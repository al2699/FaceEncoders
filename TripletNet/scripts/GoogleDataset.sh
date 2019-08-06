in_path="/data1/Alan/GoogleDataset-Test/"
out_path="/data1/Alan/GoogleDataset-Test-Processed"
for i in $(seq 0 0)
do
	./FeatureExtraction -f $in_path$i".jpg" -out_dir $out_path -simscale 1.0 -simsize 224 -format_aligned jpg
	mv $out_path$i"_aligned/frame_det_00_000001.jpg" $out_path$i".jpg"
	rm -r $out_path$i"_aligned/"
	rm $out_path*.txt $out_path*.hog
done
