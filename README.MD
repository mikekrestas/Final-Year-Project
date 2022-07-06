*Technical Documentation*

**Purpose**<br />
The purpose of this document is to provide further information regarding the “Implementing a trading strategy” web application. This document also explains in detail some of the important aspects of the application’s code and development. This technical analysis also includes a user guide as well as a description of the main trading strategy used in the demonstration.


**Technical Analysis** <br />
For the development of this web application, I decided to work with Streamlit framework which is a free web framework that helps create great visualizations of data using python. The IDE of choice is PyCharm that enabled me to easily connect my project to the capstone project repository created by the university and commit my work easily with a just a few clicks.
To break down the application in some different sections, I added a side menu on the left side of the page, in order to help the app be more efficient and also allow further future developments with no problem of over-adding features in one page. The user can choose between the “Main”, “Trading Strategy” and “News” page. I managed to achieve that by adding different if statement after creating the menu with the three different variables.
One of the very important features of the application is to enable the user to choose a stock ticker and be able to view some historical data, statistics and also a dynamic graph with a slide bar to zoom in and out. To do that, I researched and plotted tables to make the visual experience of the user as pleasant as possible. 
The news page of the application is definitely one of the essential features of the web application since I am looking to create not just a trading strategy application, but also a smart financial tool that also includes the latest news for stocks. For the news/tweets part I used the StockTwits API to get the needed data and then I ran them through a for-loop to get back only the needed attributes.
One of the main features of the program is to display the returns of a chosen stock’s trading strategy. For the MVP of the development, I have decided to work a bit with the Moving-Average-Convergence-Divergence (MACD) strategy. Although the MACD display is not dynamic at this point of the development, I have implemented code that takes the CSV-file data of a stock and then calculates the entry and exit points of stock trading, and it also displays them in a colorful graph. In the display the user can easily spot the buys/sells of the stock by using colored ^ and v sings as well as the results of the trading period.


**User Guide** <br />
The user opens up the web page and the screen loads in the main page. On the left side of the page there is a side menu that enables the user to traverse through the available pages.
Main Page: The user can insert a stock ticker in the search bar and then view a stock graph including some statistics and data about it. In order to zoom in and out of the graph, the user can use the slides of the bar on the sides or even click on the beginning of the zoomed in graph that they want and drag the mouse to the end of it.
Stock Tweets Page: if the user has previously chosen a stock to view including displaying the statistics part, the stock tweets page will automatically load the previously selected stock’s news. If there is a desire b. If there is a desire by the user to pick a different stock to watch, they can easily do that by using the available search bar on the left side menu.
Trading Strategy: the trading strategy at this part of the development loads up a pre-selected stock’s results. In the page there is a description of the used strategy and a well-structured graph including the results of the trading period.
Sign-Up and Login Pages: The users also have the ability to create accounts in the system to use from the Sign-Up page by inserting a username and a password, and later on once they have an account they can visit the Login page to log in into their account by using their credentials. A super user has been created as well, only for the use of admins to check on the activity of account creation. 


**Project Management** <br />
For developing this product, I had to make use of some useful tools provided by the university as well as being able to be checked by my supervisors. To organize my tasks and goals for this academic year, I made use of the Jira software management and through the given Kanban board by the university, I implemented all my tasks, stories, epic and bugs as well as stored all of the supervisor notes every week regarding my performance. This was a great way to keep myself organized for the final report as well, as I will be able to look back on what I did in specific dates of the year. Finally, the use of GitLab was also very important since I was able to commit my work with just a few clicks and upload it on the GitLab repository made by the university. 

**Description of Trading Strategies** <br />
As of this point, I have been working towards making a MACD trading strategy.
The Moving-Average-Convergence-Divergence (MACD) strategy is an indicator that follows the trend of the relationship between two EMA’s (exponential moving average). By subtracting the shorter EMA from the larger we get the MACD line and by using a signal line we can start trading depending on their relations. For example, when the MACD line is passing the signal line, that indicates that there is a strong buy opportunity and when the opposite happens it means there is a sell opportunity.
