import json

model_type = ["gin", "gcn", "gin_jk"] # swl_gnn
dim_list = [2,4,8,16,32,64,128,256,512,1024]
num_hops = [1,2,3,4,5,6,7,8,9,10]
# op_base = ["adj"]

for m in model_type:
    for d in dim_list:
        for h in num_hops:
            try:
                file_name = m + '_dim' + str(d) + '_hop' + str(h) + '.json'
            
            except Exception as e:
                print("Wrong here:", e, m, d, h)
                exit(0)

            setting = {
                "model": m,
                "input_dim": -1,
                "input_channel": "feat",
                "hid_dim": d,
                "output_dim": d,
                "num_classes": -1,
                "output_channel": "out",
                "num_hops": h, # for models other than swl-gnn
                "nhop_gcn": 0, # for swl
                "nhop_gin": 0, # for swl
                "nhop_min_triangle": False, # for swl
                "nhop_motif_triangle": False, # for swl
                "stack_op": False, # for swl
                "no_identity": False # for swl-gnn
            }

            with open("configs/" + m + "/" + file_name , 'w') as fp:
                json.dump(setting, fp)

model_type = ["swl_gnn"] # swl_gnn
dim_list = [2,4,8,16,32,64,128,256,512,1024]
nhop_gcn = list(range(1,10))
nhop_gin = list(range(1,10))
nhop_min_triangle = [True, False]
nhop_motif_triangle = [True, False]
stack_op = [True, False]
no_identity = [True, False]

for m in model_type:
    for d in dim_list:
        for h in nhop_gcn:
            for mint in nhop_min_triangle:
                for mott in nhop_motif_triangle:
                    for stack in stack_op:
                        for n_i in no_identity:
                            try:
                                file_name = m + '_dim' + str(d) + '_gcn' + str(h) + '_mint' + str(int(mint)) + '_mott' + str(int(mott)) + '_stack' + str(int(stack)) + '_idnt' + str(1-int(n_i)) + '.json'
                            
                            except Exception as e:
                                print("Wrong here:", m, d, h, mint, mott, stack, n_i)
                                exit(0)


                            setting = {
                                "model": m,
                                "input_dim": -1,
                                "input_channel": "feat",
                                "hid_dim": d,
                                "output_dim": d,
                                "num_classes": -1,
                                "output_channel": "out",
                                "num_hops": 0, # for models other than swl-gnn
                                "nhop_gcn": h, # for swl
                                "nhop_gin": 0, # for swl
                                "nhop_min_triangle": mint, # for swl
                                "nhop_motif_triangle": mott, # for swl
                                "stack_op": stack, # for swl
                                "no_identity": n_i # for swl-gnn
                            }

                            with open("configs/" + m + "/" + file_name , 'w') as fp:
                                json.dump(setting, fp)

        for h in nhop_gin:
            for mint in nhop_min_triangle:
                for mott in nhop_motif_triangle:
                    for stack in stack_op:
                        for n_i in no_identity:
                            try:
                                file_name = m + "_dim" + str(d) + "_gin" + str(h) + "_mint" + str(int(mint)) + "_mott" + str(int(mott)) + "_stack" + str(int(stack)) + "_idnt" + str(1-int(n_i)) + ".json"

                            except Exception as e:
                                print("Wrong here:", m, d, h, mint, mott, stack, n_i)
                                exit(0)

                            setting = {
                                "model": m,
                                "input_dim": -1,
                                "input_channel": "feat",
                                "hid_dim": d,
                                "output_dim": d,
                                "num_classes": -1,
                                "output_channel": "out",
                                "num_hops": 0, # for models other than swl-gnn
                                "nhop_gcn": 0, # for swl
                                "nhop_gin": h, # for swl
                                "nhop_min_triangle": mint, # for swl
                                "nhop_motif_triangle": mott, # for swl
                                "stack_op": stack, # for swl
                                "no_identity": n_i # for swl-gnn
                            }

                            with open("configs/" + m + "/" + file_name , 'w') as fp:
                                json.dump(setting, fp)