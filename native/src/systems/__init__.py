from .gene_regulatory import GeneRegulatorySystem
from .eco_epidemic import EcoEpidemicSystem

SYSTEM_REGISTRY = {'gene_regulatory': GeneRegulatorySystem, 'eco_epidemic': EcoEpidemicSystem}
