from agents.fac import FACAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.fbrac import FBRACAgent
from agents.rebrac import ReBRACAgent

agents = dict(
    fac=FACAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    rebrac=ReBRACAgent,
    fbrac=FBRACAgent
)
