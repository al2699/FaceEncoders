# Face Encoders

### BP4D (OpenFace pre-processed) data pipeline [all scripts under scripts/]:
1.  The data was downloaded from the FERA2015 dataset website
2.  The data was pre-processed through OpenFace using the script: BP4D.sh
3.  Once the images were done with OpenFace pre-processing. Each individual new face aligned frame came with a corresponding csv file which had all sorts of information such as head post, AUs, etc. Thus all of the CSVs were augmented together using: concatCSVs.py.
4.  Once the full OpenFace pre-processed BP4D dataset was created and the respective CSV was created, the data was cleaned in various ways:
	-	First the data was iterated through and each entry (entry row in the data csv) was checked to see if openface outputted a success status of 0 for that entry (this indicates the image was not aligned/cropped properly)
    
	-	Once the unsuccessful processed images were removed the data was put through: remove.py to remove any zero images (images which are just fully black images)
    

5.  Now that the data was all the clean, the next step was to create test/validation/train splits. The images were split as follows:
    
	-	Using create_splits.py (10% test, 10% valid, 80% training and random seed of 42), indices from the main bp4d csv were created for the respective training, testing, and validation datasets
    
	-	Once the indices were created, they were used to gather the respective rows from the main csv file using move.py in order to create the respective “train.csv”, “test.csv”, and “valid.csv” files
    

6. Next, all of the images’ respective Action Units (AUs) were added to the csvs by downloading the “BP4D-AUIntensityCodes3.0” folder off of the FERA2015 site and using the transfer.py script to move all of the AU intensity labels to the right images (Note: although the transfer.py script produces CSVs in the form of *_new.csv, they were renamed to their original names of {train.csv, test.csv, valid.csv})

### FEC (OpenFace pre-processed) data pipeline [all scripts under TripletNet/scripts/]:
1.  The initial data CSVs were downloaded from ([https://ai.google/tools/datasets/google-facial-expression/](https://ai.google/tools/datasets/google-facial-expression/))
    
2.  The images (since the CSVs only provide links) were scraped off the web using the script: get_images.py (Note: all non downloadable images’ links were pasted into dl.txt)
    
3.  Once the images were scraped into a folder, and a CSV had been generated with a mix of links and paths to the downloaded images (if the CSV provided a web link to an image instead of a local computer path then it meant that the image was unable to download due to not existing anymore), the CSV was then put through the script: remove_links.py in order to remove rows/samples with a link as one of the triplet images (the entire triplet needed to be removed because one or more of the images did not exist anymore meaning no comparison could be done)
    
4.  Next, all of the images’ paths in the main FEC CSV were tested to see if they existed by simply attempting to open every single one using: remove_non_exist.py
    
5.  Once it was confirmed that all of the scraped images existed and were openable, they were pre-processed by using OpenFace via the script: GoogleDataset.sh
    
6.  Next, a CSV of the CSVs produced by OpenFace was made using concatCSVs.py
    
7.  Then, all of the images who’s OpenFace success rating was 0 (meaning it failed to produce a face aligned/cropped image) were removed
    
8.  Next, the images were searched and any zero images (completely black images) were removed from the main dataset CSV using: remove.py
    
9.  Next indices for the respective train/validation datasets were generated using create_splits.py (seed: 42, 10% test, 9% validation, 81% train) [Note: the test set indices were not produced in this manner because the dataset already came with test set set aside]
    
10.  Finally, the final train.csv and valid.csv files were created using move.py (this script used the indices provided by create_splits.py)
