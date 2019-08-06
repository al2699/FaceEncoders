in_path="/home/ICT2000/ahernandez/BP4D-Unprocessed/"
out_path="/data/deep/Alan/FacialEncodingDataset-OpenFace/BP4D/"
#THE F PART:
for i in $(seq -f "%03g" 1 18)
do
	first=$in_path"F"$i"/"
	for j in $(seq 1 8)
	do
		second=$first"T"$j"/"
		echo second: $second
		files=$second*.jpg
		for file in $files
		do
			in_file=$file
			write_path=$out_path"F"$i"/T"$j"/"
			echo "Processing: "$in_file
			echo "Writing to: "$write_path
			./FeatureExtraction -f $in_file -out_dir $write_path -simscale 1.0 -simsize 224 -format_aligned jpg
			substr=${file:50}
			name=${substr:0:${#substr} - 4}
			echo "sub:"$name
			mv $write_path$name"_aligned/frame_det_00_000001.jpg" $write_path$name".jpg"
			rm -r $write_path$name"_aligned/"
			rm $write_path*.txt $write_path*.hog
		done
	done
done
