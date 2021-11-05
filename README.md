# PopularityPrediction-DesignLab
## Design Document
> https://docs.google.com/document/d/15OcOrzC_eUpuTd_W-arWwHjBorBUIQ9t7mSlqp3SQWg/edit?usp=sharing
## Data
Install the unrar on linux (Cent-Os Distribution in our case) :
> https://www.tecmint.com/how-to-open-extract-and-create-rar-files-in-linux/

unrar the data : 
> unrar x data.rar


# Repository Structure
- Folder **member\_level\_features**  :
  - This Folder contains member level feature for each member based on all events
  - File Name : [group\_name].csv
  - There are 12 feature cols.
  - first two are membe\_id and group\_id.
- Folder **group\_level\_features** :
  - This Folder contains group level feature for group on all events
  - File Name : [group\_name].csv
  - There are 14 feature cols.
  - First is Group ID
- File I merged all all results from  **group\_level\_features**  to single file **group\_level\_features.csv**
- Folder **NMF\_member\_level\_features** :
  - This Folder contains the **Non-negative Matrix Factorization (NMF)** of member\_level\_features based on all events.
  - Also known as Member **Role Discovery Vector/Matrix**
  - File Name : [group\_name].csv
  - There are 6 feature cols.
  - first two are membe\_id and group\_id.
- Folder **member\_level\_features\_windowwise**  :
  - This Folder contains member level feature for each member based on events in window . (Window Size : 10 (In our case))
  - File Name : [group\_name]\_[window\_number].csv
  - There are 12 feature cols.
  - first two are membe\_id and group\_id.
- Folder **NMF\_member\_level\_features\_windowwise**  :
  - This Folder contains the **Non-negative Matrix Factorization (NMF)** of member\_level\_features based on events in window . (Window Size : 10 (In our case))
  - Also known as Member **Role Discovery Vector/Matrix**
  - File Name : [group\_name]\_[window\_number].csv
  - There are 6 feature cols.
  - first two are membe\_id and group\_id.
- Folder **graph\_windowwise** :
  - This folder contains the graph on member of group who have attended atleast one event in this group
  - File Name : [group\_name]\_[window\_number].txt
  - First line of each line conatains list of nodes in graph
  - rest of the lines contain the edges in form **(From,to, Weight)**.



- Folder **group\_level\_features\_windowise**  :
  - This Folder contains group level feature for group based on events in window . (Window Size : 10 (In our case))
  - File Name : [group\_name]\_[window\_number].csv
  - There are 14 feature cols.
  - first is group ID


