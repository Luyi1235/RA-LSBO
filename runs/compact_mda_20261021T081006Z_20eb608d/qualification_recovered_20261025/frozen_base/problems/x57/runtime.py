"""Imported only inside the isolated worker after configuring local paths."""

from RCAIDE.Framework.Core import Data
import Procedure
import X57_Analysis as Vehicle_Module


def setup():
    nexus = Data()
    nexus.vehicle_configurations = Vehicle_Module.configs_setup(Vehicle_Module.vehicle_setup())
    nexus.analyses = Vehicle_Module.analyses_setup(nexus.vehicle_configurations)
    nexus.missions = Vehicle_Module.missions_setup(Vehicle_Module.mission_setup(nexus.analyses))
    nexus.procedure = Procedure.setup()
    nexus.summary = Data()
    nexus.total_number_of_iterations = 0
    return nexus
