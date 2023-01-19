# Zillow Project

## Project Description
Zillow is a real estate company that helps people find properties for sale/rent across the world. I have been tasked with creating a program that can more accurately determine the value of a home before it is posted for sale. I will only be looking at the properties that had a transaction in the year of 2017.

## Project Goals
- To discover the drivers of value of properties
- Use the drivers to develop a ML program that determines the value of properties
- Deliever a report to a technical data science team

## Questions to answer
- Is there a difference in price of the house between fips codes? If so, how much of a difference?
- Does the number of bed/bath affect the property value?
- Does when the property was built affect the value?
- Does lot size affect property value?

## Intial Thoughts and Hypothesis
I believe the main drivers behind the value of the properties is larger property size, higher number of bathrooms and bedrooms, lot size and when the property was built. Those with higher lot/property size and more bedrooms/bathrooms will be worth more. Furthermore, the properties that were built more recently will also be worth more.

## Planning
- Use the aquire.py already used in previous exerices to aquire the data necessary
- Use the code already written prior to aid in sorting and cleaning the data
- Discover main drivers
 - First identify the drivers using statistical analysis
 - Create a pandas dataframe containing all relevant drivers as columns
- Develop a model using ML to determine churn based on the top 3 drivers
 - MVP will consist of one model per driver to test out which can most accurately predict churn
 - Post MVP will consist of taking most accurate and running it through multiple models
 - Goal is to achieve at least 80% accuracy with at least one model
- Draw and record conclusions

## Data Dictionary

| Target Variable | Definition|
|-----------------|-----------|
| tax_value | The total tax assessed value of the parcel |

| Feature  | Definition |
|----------|------------|
| bed |  Number of bedrooms in home |
| bath |  Number of bathrooms in home including fractional bathrooms |
| sqft |  Calculated total finished living area of the home |
| year |  The Year the principal residence was built |
| fips | ederal Information Processing Standard code |
| lot_sqft |  Area of the lot in square feet |