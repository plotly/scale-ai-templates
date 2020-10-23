
*Source data:*

[link](https://www.kaggle.com/snap/amazon-fine-food-reviews)

### Snowflake DB Setup notes:
db created with:

    create or replace database food_reviews;select current_database(), current_schema();use food_reviews;  # for changing to database
    create or replace table food_reviews_proc 
        (id int, product_id string, user_id string, profile_name string, 
        helpful_num int, helpful_denom int, score int, review_time int, 
        summary string, text string, review_date date, ner_str_text string, 
        ner_str_summary string);
    create or replace warehouse dash_snowflake with 
        warehouse_size = 'X-SMALL'
        auto_suspend = 180
        auto_resume = true
        initially_suspended = true;

IMPORTANT: To function properly, the app requires environment variables:
- SNOWSQL_WAREHOUSE
- SNOWSQL_PWD

to be set

When importing the CSV file (amazon_fine_food_reviews_proc.csv), create a new 'filetype' in Snowflake to specify that the CSV is enclosed by double quotation marks.
See: https://community.snowflake.com/s/question/0D50Z00008zR7f0SAC/why-would-commas-within-double-quotes-cause-an-error



### Setup

First, download this dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews

Then, please create a snowflake account, and create a public warehouse named "dash_snowflake" and a database named "FOOD_REVIEWS". If you decide to use other names, please modify the variables `flake_db` and `flake_warehouse` in `app.py`. Finally, upload the CSV file to the table.


To authenticate, please find your account ID, password and username. Then, inside your `.bashrc` or `bash_profile` or `dotenv`, export your environment variables:
```
export FLAKE_ACCOUNT="..."
export FLAKE_PW="..."
export FLAKE_USER="..."
```
