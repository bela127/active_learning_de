#from ide.experiments.de_lin import blueprint
#from ide.experiments.de_sqr import blueprint

from ide.experiments.ide_lin import blueprint
#from ide.experiments.ide_sqr import blueprint

#from ide.experiments.ide_grid_blueprint import blueprint

#from ide.experiments.optimal_query_lin import blueprint
#from ide.experiments.optimal_query_sqr import blueprint

from ide.core.experiment_runner import ExperimentRunner

er = ExperimentRunner([blueprint])
er.run_experiments()
