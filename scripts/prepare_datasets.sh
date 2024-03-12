mkdir data
cd data/
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2014.zip 
unzip -q train2014.zip 

wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip --no-check-certificate
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip --no-check-certificate
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip --no-check-certificate
unzip -q refcoco.zip
unzip -q refcoco+.zip
unzip -q refcocog.zip

cd ../../
python utils/convert_refexp_to_coco.py