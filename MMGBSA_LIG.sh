#/bin/bash

runs="run1 run2 run3"
complex_ids="MD1 MD2 MD3 MD5"

for complex_id in $complex_ids
do
        for run in $runs
        do
                cd /Users/gil_f.hoben/Troponin_md_processed/${complex_id}/${run}/
                complex_dir=$PWD
                mkdir ./complete_mmgbsa
                cd ./complete_mmgbsa
                cp /Users/gil_f.hoben/Troponin_md_processed/MMGBSA/Protocols/mmpbsa.in .
                cp /Users/gil_f.hoben/Troponin_md_processed/MMGBSA/Protocols/*.prmtop .
                $AMBERHOME/bin/MMPBSA.py -O -sp troponin_egcg_nowater.prmtop -lp troponin_egcg_ligand.prmtop -rp troponin_egcg_receptor.prmtop \
                -cp troponin_egcg_complex.prmtop -i mmpbsa.in -o mmpbsa.out -y "../Trajectories/*.nc" > mmpbsa.log
        done
done