# This code is built on 'fql.py'

import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
import numpy as np

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Actor, Value, Scalar


class FACAgent(flax.struct.PyTreeNode):
    """Flow Actor-Crtic (FAC) agent."""

    rng: Any
    ac_network: Any
    bc_network: Any 
    config: Any = nonpytree_field()

    # --- AC loss -----------------------------------------------------------
    def critic_loss(self, batch, grad_params, rng, use_cp=True, online_finetuning=False):
        """Compute the flow-based conservative critic loss."""
        # Bellman backup loss
        rng, sample_rng, est_rng = jax.random.split(rng, 3) # [num_q, ]
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng) # [batch, a_dim]
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.ac_network.select('target_critic')(batch['next_observations'], actions=next_actions) # [num_q, batch]
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q # [batch, ]
        q_pred = self.ac_network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params) # [num_q, batch]
        td_loss = jnp.square(q_pred - target_q).mean()

        if use_cp:
            # Flow-based (conservative) Q penalization
            if not online_finetuning:
                if self.config['fac_threshold'] == 'batch_adaptive': # an epsilon scheme
                    estimated_logp_beta = batch['estimated_logp'] # [batch,]
                elif self.config['fac_threshold'] == 'batch_wide_constant':
                    estimated_logp_beta = jnp.full_like(batch['estimated_logp'], jnp.min(batch['estimated_logp'])) # [batch,]
                elif self.config['fac_threshold'] == 'dataset_wide_constant':
                    estimated_logp_beta = batch['estimated_logp_min'] # [batch,]
            else:
                rng, est_rng = jax.random.split(rng, 2) # [num_q, ]
                estimated_logp_beta = jax.lax.stop_gradient(self.logprob_given_actions(observations=batch['observations'],
                                                                                       actions_final=batch['actions'],
                                                                                       rng=est_rng,
                                                                                       mode=self.config['logp_method'],
                                                                                       )) # [batch,]
                if self.config['fac_threshold'] != 'batch_adaptive':
                    # Due to additional transition during online finetuning, 
                    # 'dataset_wide_constant' threshold scheme may increases unnecessary computation.
                    # Thus, we can use 'batch_wide_constant' threshold scheme.
                    estimated_logp_beta = jnp.full_like(estimated_logp_beta, jnp.min(estimated_logp_beta)) # [batch,]
            
            rng, sample_rng, est_rng = jax.random.split(rng, 3) # [num_q, ]

            penalty_actions = jnp.clip(self.sample_actions(batch['observations'], seed=sample_rng), -1, 1) # [batch, a_dim]
            qs_penalty_actions = self.ac_network.select('critic')(batch['observations'], actions=penalty_actions, params=grad_params) # [num_q, batch]
            estimated_logp_pi = jax.lax.stop_gradient(self.logprob_given_actions(observations=batch['observations'],
                                                                                 actions_final=penalty_actions,
                                                                                 rng=est_rng,
                                                                                 mode=self.config['logp_method'],
                                                                                 )) # [batch,]
            diff_logp = estimated_logp_pi - estimated_logp_beta

            if self.config['weight_type'] == 'linear':
                weight_penalty_per_actions = 1.0 - jnp.exp(diff_logp) # control below threshold
                # weight_penalty_actions = jnp.exp(diff_logp) # control above threshold
                weight_penalty_per_actions = jnp.clip(weight_penalty_per_actions, 0., 1.)
            else:
                if self.config['weight_type'] == 'logarithmic':
                    temperature_for_weight = 1.0
                elif self.config['weight_type'] == 'convex':
                    temperature_for_weight = 0.5
                elif self.config['weight_type'] == 'concave':
                    temperature_for_weight = 2.0
                weight_penalty_per_actions = 1.0 - (jnp.log1p(jnp.exp(diff_logp)**temperature_for_weight) / jnp.log(2.0))
                weight_penalty_per_actions = jnp.clip(weight_penalty_per_actions, 0., 1.)


            qs_penalty_actions = jax.lax.stop_gradient(weight_penalty_per_actions) * qs_penalty_actions # [num_q, batch]

            critic_penalty = self.config['fac_alpha'] * (qs_penalty_actions) # [num_q, ]
            critic_penalty = critic_penalty.mean()

        else:
            # if use_cp=False, no Q-value penalizing.
            critic_penalty = 0.0

        critic_loss = td_loss + critic_penalty

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q_pred.mean(),
            'q_max': q_pred.max(),
            'q_min': q_pred.min(),
            'penalty_mean': critic_penalty,
            'diff_logpi_mean': diff_logp.mean(),
            'diff_logpi_max': diff_logp.max(),
            'diff_logpi_min': diff_logp.min(),
            'est_logpi_mean': estimated_logp_pi.mean(),
            'est_logpi_max': estimated_logp_pi.max(),
            'est_logpi_min': estimated_logp_pi.min(),
            'est_logbeta_mean': estimated_logp_beta.mean(),
            'est_logbeta_max': estimated_logp_beta.max(),
            'est_logbeta_min': estimated_logp_beta.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FAC actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        if self.config['distil_loss']:
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.ac_network.select('actor_onestep')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
        else:
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            actor_actions = self.ac_network.select('actor_onestep')(batch['observations'], noises, params=grad_params)
            distill_loss = 0.0

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.ac_network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss
        actor_loss = self.config['fac_lambda'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
        }
    # --------------------------------------------------------------
    
    # --- BC loss -----------------------------------------------------------
    def bc_loss(self, batch, grad_params, rng):
        """Compute the flow proxy (behavior cloning) loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.bc_network.select('bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }
    # --------------------------------------------------------------

    # --- Update -----------------------------------------------------------
    @partial(jax.jit, static_argnames=("mode", "use_cp", "online_finetuning"))
    def total_loss(self, batch, grad_params, 
                   mode,                    # mode: 'train_bc' or 'train_ac'
                   use_cp=True,             # if training Q-values conservatively or not for online finetuning
                   online_finetuning=False, # if online finetuning or not
                   rng=None):
        """Compute the AC or BC loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        if mode == 'train_bc':
            rng, bc_rng = jax.random.split(rng, 2)
            bc_loss, bc_info = self.bc_loss(batch, grad_params, bc_rng)
            for k, v in bc_info.items():
                info[f'bc/{k}'] = v

            loss = bc_loss

        elif mode == 'train_ac':
            rng, actor_rng, critic_rng = jax.random.split(rng, 3)
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, use_cp, online_finetuning)
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v

            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            loss = critic_loss + actor_loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=("mode", "use_cp", "online_finetuning"))
    def update(self, batch, mode,       # mode: 'train_bc' or 'train_ac'
               use_cp=True,             # if training Q-values conservatively or not for online finetuning
               online_finetuning=False, # if online finetuning or not
               ):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, mode, use_cp, online_finetuning, rng=rng)

        if mode == 'train_bc':
            new_bc_network, info = self.bc_network.apply_loss_fn(loss_fn=loss_fn)

            return self.replace(bc_network=new_bc_network, rng=new_rng), info
        
        elif mode == 'train_ac':
            new_ac_network, info = self.ac_network.apply_loss_fn(loss_fn=loss_fn)
            self.target_update(new_ac_network, 'critic')

            return self.replace(ac_network=new_ac_network, rng=new_rng), info
    # --------------------------------------------------------------

    # --- Action sampling-----------------------------------------------------------
    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)

        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        ) # [batch, a_dim]
        actions = self.ac_network.select('actor_onestep')(observations, noises) # [batch, a_dim]
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.bc_network.select('bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.bc_network.select('bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    # --------------------------------------------------------------
    

    # --- Compute log-denstiy (logp) -----------------------------------------------------------
    @staticmethod
    def make_divergence_exact(v_apply):
        """v_apply: (obs[B,D_s], act[B,D_a], t[B,1]) -> vel[B,D_a]"""
        def single_div(o, a, t):
            # single-sample v_fn: R^A -> R^A
            def v_fn(a_in):
                return v_apply(o[None, :], a_in[None, :], t[None, None])[0]
            J = jax.jacrev(v_fn)(a) # Jacobian; (A, A)
            return jnp.trace(J)
        return jax.vmap(single_div, in_axes=(0, 0, 0))

    @staticmethod
    def make_divergence_hutchinson(v_apply, probes=8, gaussian=False):
        """Hutchinson trace estimator: E[e^T J e], e ~ normal or Rademacher distribution"""
        def single_div(o, a, t, key):
            def v_fn(a_in):
                return v_apply(o[None, :], a_in[None, :], t[None, None])[0]

            def vJv(k):
                _, vjp = jax.vjp(v_fn, a) # vector-Jacobian product, i.e., eps^T J
                eps = (jax.random.normal(k, a.shape, dtype=a.dtype) if gaussian
                    else jax.random.rademacher(k, a.shape, dtype=a.dtype))
                return jnp.dot(vjp(eps)[0], eps) # eps^T J eps

            keys = jax.random.split(key, probes)
            vals = jax.vmap(vJv)(keys)
            return vals.mean()

        return jax.vmap(single_div, in_axes=(0, 0, 0, 0))
    
    @partial(jax.jit, static_argnames=("mode"))
    def logprob_given_actions(
        self,
        observations,   # (B, S_dim)
        actions_final,  # (B, A_dim)
        rng=None,       # for Hutchinson estimation
        mode="exact",   # "exact" or "hutch"
    ):
        """Return: logp: (B,); log-density of flow BC model"""
        N = self.config['flow_steps']
        dt = 1.0 / N
        B, A_dim = actions_final.shape

        if self.config['encoder'] is not None:
            obs_enc = self.bc_network.select('bc_flow_encoder')(observations)
        else:
            obs_enc = observations

        def v_apply(obs_b, act_b, t_b): # obs_b: (B, S_dim), act_b: (B, A_dim), t_b: (B,1)
            return self.bc_network.select('bc_flow')(obs_b, act_b, t_b, is_encoded=True)

        # 1) Define divergence
        if mode == "exact":
            div_fn = self.make_divergence_exact(v_apply)
            need_key = False
        elif "hutch" in mode:
            assert rng is not None
            if mode == "hutch-rade":
                div_fn = self.make_divergence_hutchinson(v_apply, 
                                                         probes=self.config['logp_hutch_probes'], 
                                                         gaussian=False)
            elif mode == "hutch-gaus":
                div_fn = self.make_divergence_hutchinson(v_apply, 
                                                         probes=self.config['logp_hutch_probes'], 
                                                         gaussian=True)
            need_key = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2) Initial point for reverse ODE problem
        a_t = actions_final # at t=1
        logp_acc = jnp.zeros((B,), dtype=actions_final.dtype) # initial point: log pi(a_1|s)=0

        if need_key:
            rngs = jax.random.split(rng, N * B)
            rngs = rngs.reshape(N, B, 2)  # (N,B,2)
        else:
            rngs = None

        # 3) Reverse ODE solve via Euler method
        def body_fun(k, carry):
            a_cur, logp_cur = carry
            # time index: from 1 to 0, so t_k = (k+1)/N
            t_k = (N - k) / N
            t_batch = jnp.full((B, 1), t_k, dtype=a_cur.dtype)

            # velocity at current (obs_enc, a_cur, t_k)
            vels = v_apply(obs_enc, a_cur, t_batch)

            # divergence
            if need_key:
                div_keys = rngs[k, :, :]  # (B, 2)
                divs = div_fn(obs_enc, a_cur, t_batch.squeeze(-1), div_keys)
            else:
                divs = div_fn(obs_enc, a_cur, t_batch.squeeze(-1))

            # update logp, a_u
            logp_new = logp_cur + divs * dt # actually, data_logp = base_logp - int_logp_div
                                            # this function returns base_logp - sum_logp_div, 
                                            # so sum_logp_div can be computed like that.
            a_new   = a_cur - vels * dt     # reverse Euler step

            return (a_new, logp_new)

        # lax.fori_loop(0, N, ...): k=0..N-1
        a_0, logp_div = jax.lax.fori_loop(0, N, body_fun, (a_t, logp_acc))

        # 4) base log prob at t=0 (assume standard normal distribution)
        base_logp = -0.5 * jnp.sum(a_0**2, axis=-1) - 0.5 * A_dim * jnp.log(2 * jnp.pi)

        return base_logp - logp_div # (B,)
    # --------------------------------------------------------------

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng_ac, init_rng_bc = jax.random.split(rng, 3)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['bc_flow'] = encoder_module()
            encoders['actor_onestep'] = encoder_module()

        # Define AC networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_critics'],
            encoder=encoders.get('critic'),
        )

        actor_onestep_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep'),
        )

        ac_network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_onestep=(actor_onestep_def, (ex_observations, ex_actions)),
        )

        ac_networks = {k: v[0] for k, v in ac_network_info.items()}
        ac_network_args = {k: v[1] for k, v in ac_network_info.items()}

        ac_network_def = ModuleDict(ac_networks)
        ac_network_tx = optax.adam(learning_rate=config['lr'])
        ac_network_params = ac_network_def.init(init_rng_ac, **ac_network_args)['params']
        ac_network = TrainState.create(ac_network_def, ac_network_params, tx=ac_network_tx)

        ac_params = ac_network.params
        ac_params['modules_target_critic'] = ac_params['modules_critic']

        # Define BC network
        bc_flow_def = ActorVectorField(
            hidden_dims=config['bc_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['bc_layer_norm'],
            encoder=encoders.get('bc_flow'),
        )
        bc_network_info = dict(
            bc_flow=(bc_flow_def, (ex_observations, ex_actions, ex_times)),
        )

        if encoders.get('bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            bc_network_info['bc_flow_encoder'] = (encoders.get('bc_flow'), (ex_observations,))

        bc_networks = {k: v[0] for k, v in bc_network_info.items()}
        bc_network_args = {k: v[1] for k, v in bc_network_info.items()}

        bc_network_def = ModuleDict(bc_networks)
        bc_network_tx = optax.adam(learning_rate=config['lr_bc'])
        bc_network_params = bc_network_def.init(init_rng_bc, **bc_network_args)['params']
        bc_network = TrainState.create(bc_network_def, bc_network_params, tx=bc_network_tx)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, ac_network=ac_network, bc_network=bc_network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fac',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate for AC.
            lr_bc=3e-4, # LR for BC_flow_model
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            bc_hidden_dims=(512, 512, 512, 512),  # BC flow network hidden dimensions.
            value_layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            bc_layer_norm=False,  # Whether to use layer normalization for the bc flow.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            distil_loss=True, # if the distillation loss is in the policy loss
            fac_alpha=1.0, # critic penalty coefficient
            fac_lambda=1.0,  # actor regularization coefficient
            fac_threshold='batch_adaptive', # use logp evaluations of the offline dataset; 'dataset_wide_constant' 'batch_wide_constant' 'batch_adaptive'
            logp_method='exact', # 'exact' or 'hutch-rade' or 'hutch-gaus'
            logp_hutch_probes=8,
            num_critics=2,
            discount=0.995,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for target Q values; 'min' or 'mean'
            weight_type='linear', # 'linear' or 'logarithmic' 'convex' 'concave'
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
