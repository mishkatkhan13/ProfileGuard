This Instagram Fake Account Detection system is designed to determine whether an Instagram profile might be fake or genuine. The system utilizes various machine learning models to predict the authenticity based on features derived from the user's profile such as profile picture presence, username length, biography length, and other relevant attributes.
Features
The system analyses the following features from an Instagram profile to predict its authenticity:
userFollowerCount (followers)
userFollowingCount (following)
userBiographyLength (comment6)
userMediaCount
userHasProfilePic (1 in input_features)
userIsPrivate (comment8)
usernameDigitCount
usernameLength
Setup Instructions
This project utilizes machine learning to detect fake accounts on Instagram based on profile characteristics. It uses Flask for the backend server, Instaloader for fetching Instagram profile data, and scikit-learn for machine learning models.
## Project Setup
### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
### Installing Dependencies
To set up the necessary environment and install all required packages, follow these steps:
1. Clone the repository:
  git clone <repository-url>

2. Navigate to the project directory:
  cd <project-directory>

3. Create a virtual environment:
  python -m venv env

And then Activate the virtual environment:
On Windows:
env\Scripts\activate

On macOS and Linux: 
source env/bin/activate


4. Install the required packages:
pip install Flask
pip install pandas
pip install numpy
pip install scikit-learn
pip install instaloader
pip install requests

5. Prepare the Dataset: Ensure that your dataset file insta_train.csv is in the project directory or adjust the path in the script accordingly.
6. Start the Flask Application: Run the Flask application with:
python app.py

How to Use
 Open your web browser and navigate to http://127.0.0.1:5000/.
 You will see a form where you can enter the username of the Instagram profile you want to check.
 After submitting the username, the system will process the profile data and return a prediction on whether the profile is likely fake or genuine.
Technologies Used
  Python: Main programming language used for the project.
  Flask: Web framework for building the web application.
 Pandas and NumPy: For data manipulation and calculations.
 Scikit-Learn: For machine learning model training and predictions.
Instaloader: For fetching Instagram profile data.
SVM and Random Forest Classifiers: Used for making predictions based on the profile features.
