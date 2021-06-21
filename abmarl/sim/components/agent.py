import numpy as np

from abmarl.sim import PrincipleAgent, ActingAgent, ObservingAgent


# ------------------ #
# --- Base Agent --- #
# ------------------ #


# --------------------- #
# --- Communication --- #
# --------------------- #



class BroadcastObservingAgent(ObservingAgent, ComponentAgent): pass


# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class LifeObservingAgent(ObservingAgent, ComponentAgent): pass
class HealthObservingAgent(ObservingAgent, ComponentAgent): pass


# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #


class PositionObservingAgent(ObservingAgent, ComponentAgent): pass


# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #


# ------------ #
# --- Team --- #
# ------------ #

class TeamObservingAgent(ObservingAgent, ComponentAgent): pass
