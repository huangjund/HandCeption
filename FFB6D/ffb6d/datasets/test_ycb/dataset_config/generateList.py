import json
import os
import numpy as np
import sys


def generate_data(config):
    output_dir, num_files, num_test, num_train, seed = (
        config["output_dir"], config["num_files"], config["num_test"], config["num_train"], config["seed"])
    os.makedirs(output_dir, exist_ok=True)

    all_terms = [f"data_syn/{i:06d}" for i in range(num_files)]
    # Shuffle terms with NumPy
    # np.random.seed(seed)
    # np.random.shuffle(all_terms)
    test_terms = all_terms[:num_test]
    train_terms = all_terms[num_test:num_test + num_train]

    with open(os.path.join(output_dir, "test_data_list.txt"), "w") as test_file:
        test_file.write("\n".join(test_terms) + "\n")

    with open(os.path.join(output_dir, "train_data_list.txt"), "w") as train_file:
        train_file.write("\n".join(train_terms) + "\n")

    print(f"Generated test.txt with {num_test} terms and train.txt with {num_train} terms in {output_dir}")


if __name__ == "__main__":
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    generate_data(config)
