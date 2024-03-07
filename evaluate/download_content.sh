mkdir data 

cd data 

wget https://www.dropbox.com/s/qeljw3vjnc416bs/table_003_cracker_box_dope_results.zip
wget https://www.dropbox.com/s/mn2yqflc6fcqaic/table_003_cracker_box.zip

unzip table_003_cracker_box_dope_results.zip 
rm table_003_cracker_box_dope_results.zip
mkdir table_dope_results/
mv table_003_cracker_box table_dope_results/scene1/

unzip table_003_cracker_box.zip
rm table_003_cracker_box.zip
mkdir table_ground_truth/
mv table_003_cracker_box table_ground_truth/scene1/

cd ../

mkdir content
cd content

wget https://www.dropbox.com/s/b61es9q5nhwtooi/003_cracker_box.zip
unzip 003_cracker_box.zip
rm 003_cracker_box.zip
mv 003_cracker_box 003_cracker_box_16k

cd ../