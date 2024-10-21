import unittest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_text

from core.utils.research.data.collect.analysis import RunnerStatsTrainer, SVRModel, Model
from core.utils.research.data.collect.analysis.datapreparer import RunnerStatsDataPreparer
from core.utils.research.data.collect.analysis.models import XGBoostModel, RidgeModel, LassoModel
from core.utils.research.data.collect.analysis.models.decision_tree_model import DecisionTreeModel


class TrainerTest(unittest.TestCase):

	def setUp(self):
		columns_size = 8
		omitted_columns = [2]
		columns = [i for i in range(columns_size) if i not in omitted_columns]

		self.bounds = (-5, 5)

		self.column_names = [
			col
			for i, col in enumerate([
				"nn.CrossEntropyLoss()",
				"ProximalMaskedLoss",
				"MeanSquaredClassError",
				"ReverseMAWeightLoss(window_size=10, softmax=True)",
				"PredictionConfidenceScore(softmax=True)",
				"OutputClassesVariance(softmax=True)",
				"OutputBatchVariance(softmax=True)",
				"OutputBatchClassVariance(softmax=True)",
			])
			if i not in omitted_columns
		]

		self.datapreparer = RunnerStatsDataPreparer(
			bounds=self.bounds,
			columns=columns,
			min_sessions=1
		)
		self.X, self.y = self.datapreparer.prepare()

		self.column_names = [
			"nn.CrossEntropyLoss()",
			"ProximalMaskedLoss",
			# "MeanSquaredClassError",
			"ReverseMAWeightLoss(window_size=10, softmax=True)",
			"PredictionConfidenceScore(softmax=True)",
			"OutputClassesVariance(softmax=True)",
			"OutputBatchVariance(softmax=True)",
			"OutputBatchClassVariance(softmax=True)",
		]

	def __visualize_result(self, model: Model, detailed=False, name=None):
		y_hat = model.predict(self.X)

		if detailed:
			self.__plot_predictions(self.X, y_hat, self.y)

		plt.figure()
		plt.plot(self.bounds, self.bounds, color='black')
		plt.axvline(x=0, color="black")
		plt.axhline(y=0, color="black")
		plt.scatter(y_hat, self.y)
		if name is not None:
			plt.title(name)

	def __plot_predictions(self, X: np.ndarray, y_hat: np.ndarray, y: np.ndarray = None):
		for i in range(X.shape[1]):
			plt.figure()
			plt.title(self.column_names[i])
			plt.axhline(y=0, color="black")
			if y is not None:
				plt.scatter(X[:, i], y)
			plt.scatter(X[:, i], y_hat)

	def __test_model(self, model, epochs=1, detailed=False, name=None):
		trainer = RunnerStatsTrainer(data_preparer=self.datapreparer)

		result = trainer.start(model, epochs=epochs)

		print(f"Result {result}")

		self.__visualize_result(model, detailed=detailed, name=name)

	def test_svr(self):
		model = SVRModel(
			kernel='rbf',
			C=10**5,
			gamma=10**5
		)
		self.__test_model(model)
		coef = model.model.dual_coef_
		for i in range(len(coef)):
			print(f"Feature {self.column_names[i]}: {coef[i]}")


		plt.show()

	def test_optimal_svr_config(self):
		KERNELS = ['linear']
		Cs = [10**i for i in range(2, 6)]
		gammas = [10**i for i in range(-2, 2)]

		for kernel in KERNELS:

			for C in Cs:
				for gamma in gammas:
					print(f"Processing Kernel: {kernel}, C: {C}, gamma: {gamma}")
					model = SVRModel(kernel=kernel, C=C, gamma=gamma)
					self.__test_model(model, name=f"Kernel={kernel}, C={C}, gamma={gamma}")

		plt.show()

	def test_decision_tree(self):
		model = DecisionTreeModel(max_depth=4)
		self.__test_model(model, detailed=True)

		rules = export_text(model.model, feature_names=self.column_names)

		tree = model.model.tree_
		n_node_samples = tree.n_node_samples

		rules_lines = rules.split("\n")
		for i, line in enumerate(rules_lines):
			if i < len(n_node_samples):
				print(f"{line} (instances: {n_node_samples[i]})")
			else:
				print(line)
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
		model = RidgeModel(alpha=0)
		self.__test_model(model, detailed=True)
		coef = model.model.coef_
		for i in range(len(coef)):
			print(f"Feature {self.column_names[i]}: {coef[i]}")

		plt.show()

	def test_optimize_alpha(self):

		alphas = [(0.1)*i for i in range(10)]
		for alpha in alphas:
			print(f"Using Alpha: {alpha}")
			model = RidgeModel(alpha=alpha)
			self.__test_model(model)

		plt.show()

	def test_lasso_model(self):
		model = LassoModel(alpha=0)
		self.__test_model(model, detailed=True)
		coef = model.model.coef_
		for i in range(len(coef)):
			print(f"Feature {self.column_names[i]}: {coef[i]}")

		plt.show()

	def test_optimize_lasso_alpha(self):

		alphas = [(0.1)*i for i in range(10)]
		for alpha in alphas:
			print(f"Using Alpha: {alpha}")
			model = LassoModel(alpha=alpha)
			self.__test_model(model, name=f"Alpha={alpha}")

		plt.show()

	def test_optimal_model(self):
		models = [
			SVRModel(),
			DecisionTreeModel(max_depth=4),
			XGBoostModel(),
			RidgeModel(alpha=0),
			LassoModel(alpha=0),
		]

		for model in models:
			print(f"Processing {model.__class__.__name__}")
			self.__test_model(model, name=model.__class__.__name__)

		plt.show()

	def test_predict_unseen(self):
		model = XGBoostModel()
		self.__test_model(model, detailed=True)

		X, y = self.datapreparer.prepare(
			runlive_tested=False,
			loss_evaluated=True,
			min_sessions=0
		)
		y_hat = model.predict(X)

		self.__plot_predictions(X, y_hat)
		plt.show()
