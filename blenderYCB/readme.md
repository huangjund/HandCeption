generate URDF from solidworks

```angular2html
mkdir output
mkdir data
mkdir data_syn

# change configs files before below steps
cd data
ln -s /ssd2/T-DexCO\ Hand/objs_custom_data/ . # link the obj models to the data folder
ln -s /ssd2/T-DexCO\ Hand/AssemV_camera/ . # link the urdf folder to the data folder
ln -s /ssd2/sun2012pascalformat/ . # link the background images to the data folder

# create a conda environment
conda create -n render python=3.10 -y

```


blenderproc run scripts/generateRendering.py