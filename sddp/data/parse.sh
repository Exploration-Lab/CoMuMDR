###############################################################################
# Stac and Molweni
###############################################################################
python build_relation_database.py --dir_path stac
cp stac/relation_database.json molweni/relation_database.json

python parse.py --dir_path stac --num_contexts 20 $1
python parse.py --dir_path molweni --num_contexts 14 $1

python parse_test.py --dir_path stac --num_contexts 37 $1 --mode dev $1
python parse_test.py --dir_path stac --num_contexts 37 $1
python parse_test.py --dir_path molweni --num_contexts 14 --mode dev $1
python parse_test.py --dir_path molweni --num_contexts 14 $1

###############################################################################
# CoMuMDR
# We need to run the build_relation_database.py script first to create the 
# relation database as CoMuMDR has a different set of relations.
###############################################################################
python build_relation_database.py --dir_path comumdr

python parse.py --dir_path comumdr --num_contexts 20 $1

python parse_test.py --dir_path comumdr --num_contexts 37 $1 --mode dev $1
python parse_test.py --dir_path comumdr --num_contexts 37 $1

###############################################################################
