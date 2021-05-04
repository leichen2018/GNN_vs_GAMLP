# GNN_vs_GAMLP
The official implementation of ''On Graph Neural Networks versus Graph-Augmented MLPs'' ICLR 2021

Authors: [Lei Chen](https://leichen2018.github.io)\*, [Zhengdao Chen](https://cims.nyu.edu/~chenzh/)\*, [Joan Bruna](https://cims.nyu.edu/~bruna/).

## Dependencies

Core packages:
```
pytorch 1.5.0
dgl 0.4.3
```

## Scripts

### Community detection

<details>
<summary>Click</summary>
<p>

* GA-MLP-H (hardness rank 1)
```
python3.7m main_sbm.py --model swl_gnn_dim10_bh120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 4.5 --q 1.5 --n-nodes 250 --n-communities 2
```

* GA-MLP-H (hardness rank 2)
```
python3.7m main_sbm.py --model swl_gnn_dim10_bh120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 4.73 --q 1.27 --n-nodes 250 --n-communities 2
```

* GA-MLP-H (hardness rank 3)
```
python3.7m main_sbm.py --model swl_gnn_dim10_bh120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 5 --q 1 --n-nodes 250 --n-communities 2
```

* GA-MLP-H (hardness rank 4)
```
python3.7m main_sbm.py --model swl_gnn_dim10_bh120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 5.5 --q 0.5 --n-nodes 250 --n-communities 2
```

* GA-MLP-H (hardness rank 5) 
```
python3.7m main_sbm.py --model swl_gnn_dim10_bh120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 6 --q 0 --n-nodes 250 --n-communities 2
```

* GA-MLP-A (hardness rank 1)
```
python3.7m main_sbm.py --model swl_gnn_dim10_gin120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 4.5 --q 1.5 --n-nodes 250 --n-communities 2
```

* GA-MLP-A (hardness rank 2)
```
python3.7m main_sbm.py --model swl_gnn_dim10_gin120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 4.73 --q 1.27 --n-nodes 250 --n-communities 2
```

* GA-MLP-A (hardness rank 3)
```
python3.7m main_sbm.py --model swl_gnn_dim10_gin120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 5 --q 1 --n-nodes 250 --n-communities 2
```

* GA-MLP-A (hardness rank 4)
```
python3.7m main_sbm.py --model swl_gnn_dim10_gin120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 5.5 --q 0.5 --n-nodes 250 --n-communities 2
```

* GA-MLP-A (hardness rank 5) 
```
python3.7m main_sbm.py --model swl_gnn_dim10_gin120_mint0_mott0_stack1_idnt1 --verbose --lr 0.001 --n-graphs 6000 --p 6 --q 0 --n-nodes 250 --n-communities 2
```

</p>
</details>

### Counting attributed walks

<details>
<summary>Click</summary>
<p>

* GA-MLP-A (Cora)
```
python3.7m main.py --dataset cora_path --lr 0.01 --epochs 20000 --num_hops 2 --hid_dim 256 --model swl_gnn --op_base adj
```

* GA-MLP-A+ (Cora)
```
python3.7m main.py --dataset cora_path --lr 0.01 --epochs 20000 --num_hops 4 --hid_dim 32 --model swl_gnn --op_base adj
```

* GIN (Cora)
```
python3.7m main.py --dataset cora_path --lr 0.01 --epochs 20000 --num_hops 2 --hid_dim 32 --model gin_jk
```

* GA-MLP-A (RRG)
```
python3.7m main.py --dataset regular --lr 0.01 --epochs 20000 --num_hops 3 --hid_dim 32 --model swl_gnn --op_base adj
```

* GA-MLP-A+ (RRG)
```
python3.7m main.py --dataset regular --lr 0.001 --epochs 20000 --num_hops 6 --hid_dim 256 --model swl_gnn --op_base adj
```

* GIN (RRG)
```
python3.7m main.py --dataset regular --lr 0.005 --epochs 20000 --num_hops 3 --hid_dim 16 --model gin_jk
```

</p>
</details>

