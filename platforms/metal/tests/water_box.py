from openmm import *
from openmm.app import *
from openmm.unit import *
import sys

Platform.loadPluginsFromDirectory(Platform.getDefaultPluginsDirectory())

width = float(sys.argv[1])
ff = ForceField('tip3pfb.xml')
modeller = Modeller(Topology(), [])
modeller.addSolvent(ff, boxSize=Vec3(width, width, width)*nanometers)
#system = ff.createSystem(modeller.topology,nonbondedMethod=NoCutoff)#PME,nonbondedCutoff=0.5)
system = ff.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0)
print(system.getNumParticles())
integrator = LangevinMiddleIntegrator(300, 1.0, 0.004)
platform = Platform.getPlatformByName('HIP')
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
temperature=True, elapsedTime=True, speed=True))
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy(tolerance=100*kilojoules_per_mole/nanometer)
simulation.step(5000)

