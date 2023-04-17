# SVM

### What is in this repo

1. src (the code we are running)
	* final.py
		- Python file with SVM implementations
			- SVM from scratch functions:
				- get_subgradient
				- svm_sgd
				- get_predictions
				- get_accuracy
				- get_loss
			- SVM from scikit functions:
				- scikit_accuracy
				- scikit_loss
				- scikit_svm\_linear
				- scikit\_svm_non\_linear
			- Cross-validation functions:
				- svm\_cross\_validation
				- scikit\_svm\_cross\_validation

2. scripts (helper code)
	* util.py
		- Python file with the following helper functions:
			- split_data
			- divide\_k\_folds
			- get\_X\_y_data
			- sgn
	* xor.py
		- Python file that generates xor.csv
	* xor_plot.py
		- Python file that generates color map that shows the decision function learned by the SVC on the XOR data

3. data (datasets)
	* data\_banknote_authentication.csv
		- Linearly separable data (from shared class folder)
	* xor.csv
		- Sample dataset that looks like XOR

4. final.pdf (final report)


### How to run the code

- Open a terminal, navigate to the `/final/src` directory, and run the project with `python3 final.py`
