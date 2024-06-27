# Insurance-price-prediction
Here we calculate the charges of insurance based on some features.

Project Overview
The Insurance Price Prediction project is designed to predict the insurance charges for individuals based on their demographic and health-related information. This project utilizes various machine learning models to provide accurate predictions and showcases the importance of data preprocessing, feature engineering, and model evaluation in machine learning pipelines.

#**Table of Contents**

Project Overview
Dataset
Project Structure
Installation
Usage
Models Used
Results
Visualization
Future Work
Contributing
License
Dataset


#The dataset used in this project is the Insurance dataset, which contains the following features:

Age
Sex
BMI (Body Mass Index)
Children (Number of children/dependents)
Smoker (Yes/No)
Region (Region in the US)
Charges (Medical charges billed by health insurance)
Project Structure
arduino
Copy code

#The structure of project is 

Insurance-Price-Prediction/
│
├── templates/
│   ├── login.html
│   ├── predict.html
│
├── static/
│   ├── images/
│   │   ├── login_image.png
│   │   ├── main_image.png
│
├── insurance_price_prediction.pkl
├── app.py
├── requirements.txt
└── README.md


#Installation

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/Insurance-Price-Prediction.git
cd Insurance-Price-Prediction
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Run the Flask application:

bash
Copy code
python app.py
Open your browser and go to http://127.0.0.1:5000/.

You will see the login page where you can enter your credentials. The default credentials are:

Username: admin
Password: 123
You can also sign up with a new username and password. The credentials will be saved and used for subsequent logins.

Once logged in, you will be redirected to the prediction page where you can input the features required for the prediction.

The model will predict the insurance charges based on the input features and display the result.


#Models Used

The project explores multiple machine learning models to predict insurance charges:

Linear Regression
Lasso Regression
Ridge Regression
K-Nearest Neighbors (KNN)
Support Vector Regressor (SVR)
Decision Tree Regressor
Random Forest Regressor
Among these, the Random Forest Regressor was found to be the most accurate model with an accuracy of approximately 87.63%.

#Results
The Random Forest Regressor model provided the best results for predicting insurance charges with the following accuracy:

Random Forest Regressor: 87.63%
Visualization
The project includes several visualizations to understand the distribution and relationships within the data:

Count plots for categorical features
Distribution plot for the target variable (charges)
Bar plots for the relationship between categorical features and charges
Scatter plots for the relationship between numerical features (BMI, age) and charges
Future Work
Implementing more advanced machine learning models and techniques such as Gradient Boosting, XGBoost, or neural networks.
Incorporating more features into the dataset to improve model accuracy.
Enhancing the user interface for better user experience.
Implementing user authentication with more robust security measures.
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License - see the LICENSE file for details.

