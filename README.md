# KaggleProject_HotelBookingDemand

## Dataset
The dataset is used in the project is Hotel Booking Demand  https://www.kaggle.com/jessemostipak/hotel-booking-demand. This data set contains booking information for a city hotel and a resort hotel and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.
## Objective
The Objective of this project is to predict whether the hotel reservation will be cancelled. The response value is binary classification value composed of 0 and 1. 0 represents the hotel booking is not cancelled, 1 represents the hotel booking is cancelled.
The audiences of the project are the hotels that want to allocate resources reasonably according to the cancellation rate of bookings during peak seasons, or the hotels that want to find out which factors cause booking cancelation. They can make improvements to these factors.
## Model Selection
The algorithms used in the project are Support Vector Machines (SVM) and the Artificial Neural Networks (ANN). There are five reasons for selecting these two algorithms: 1. The response variable is a categorical variable, SVM and ANN can be used for classification problems. 2. The size of the dataset is not large enough. 3. There are categorical and numerical predictors.  4. SVM and ANN have the ability to learn and model non-linear and complex relationships 5. Predictors importance can be computed. 
## Model Assessment
The double 10-fold cross validation method is used for model assessment. 
