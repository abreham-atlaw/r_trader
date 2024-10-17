import unittest

import matplotlib.pyplot as plt
from sklearn.tree import export_text

from core.utils.research.data.collect.analysis import RunnerStatsTrainer, SVRModel, Model
from core.utils.research.data.collect.analysis.datapreparer import RunnerStatsDataPreparer
from core.utils.research.data.collect.analysis.models import XGBoostModel, RidgeModel, LassoModel
from core.utils.research.data.collect.analysis.models.decision_tree_model import DecisionTreeModel


class TrainerTest(unittest.TestCase):

	def setUp(self):
		self.datapreparer = RunnerStatsDataPreparer(
			bounds=(-5, 5)
		)
		self.X, self.y = self.datapreparer.prepare()

		self.column_names = [
			"nn.CrossEntropyLoss()",
			"ProximalMaskedLoss",
			"MeanSquaredClassError",
			"ReverseMAWeightLoss(window_size=10, softmax=True)",
			"PredictionConfidenceScore(softmax=True)",
			"OutputClassesVariance(softmax=True)",
			"OutputBatchVariance(softmax=True)",
			"OutputBatchClassVariance(softmax=True)",
		]

	def __visualize_result(self, model: Model, detailed=False, name=None):
		y_hat = model.predict(self.X)

		if detailed:
			for i in range(self.X.shape[1]):
				plt.figure()
				plt.title(self.column_names[i])
				plt.scatter(self.X[:, i], self.y)
				plt.scatter(self.X[:, i], y_hat)

		plt.figure()
		plt.scatter(y_hat, self.y)
		if name is not None:
			plt.title(name)

	def __test_model(self, model, epochs=1, detailed=False, name=None):
		trainer = RunnerStatsTrainer()

		result = trainer.start(model, epochs=epochs)

		print(f"Result {result}")

		self.__visualize_result(model, detailed=detailed, name=name)

	def test_svr(self):
		model = SVRModel()
		self.__test_model(model)
		plt.show()

	def test_optimal_kernel(self):
		KERNELS = ['rbf', 'linear', 'poly']

		for kernel in KERNELS:

			model = SVRModel(kernel=kernel)
			self.__test_model(model)

		plt.show()

	def test_decision_tree(self):
		model = DecisionTreeModel(max_depth=4)
		self.__test_model(model, detailed=True)
		rules = export_text(model.model, feature_names=self.column_names)
		print(rules)
		plt.show()

	def test_dt_optimal_depth(self):

		depths = [1, 2, 3, 4, 5, 6]

		for depth in depths:
			print(f"Processing Depth: {depth}")
			model = DecisionTreeModel(max_depth=depth)
			self.__test_model(model, name=f"Depth={depth}")

		plt.show()

	def test_xgboost(self):
		model = XGBoostModel()
		self.__test_model(model)
		importances = model.model.feature_importances_
		sorted_idx = importances.argsort()

		for i in sorted_idx:
			print(f"Feature {self.column_names[i]}: {importances[i]}")

		booster = model.model.get_booster()
		importance = booster.get_score(
			importance_type='weight'
		)
		print(importance)

		plt.show()

	def test_ridge_model(self):
		model = RidgeModel()
		self.__test_model(model, detailed=True)
		coef = model.model.coef_
		for i in range(len(coef)):
			print(f"Feature {self.column_names[i]}: {coef[i]}")

		plt.show()

	def test_optimize_alpha(self):

		alphas = [(0.1)**i for i in range(1, 6)]
		for alpha in alphas:
			print(f"Using Alpha: {alpha}")
			model = RidgeModel(alpha=alpha)
			self.__test_model(model)

		plt.show()

	def test_lasso_model(self):
		model = LassoModel()
		self.__test_model(model, detailed=True)
		coef = model.model.coef_
		for i in range(len(coef)):
			print(f"Feature {self.column_names[i]}: {coef[i]}")

		plt.show()

	def test_optimal_model(self):
		models = [
			SVRModel(),
			DecisionTreeModel(max_depth=3),
			XGBoostModel(),
			RidgeModel(),
			LassoModel(),
		]

		for model in models:
			print(f"Processing {model.__class__.__name__}")
			self.__test_model(model, name=model.__class__.__name__)

		plt.show()
