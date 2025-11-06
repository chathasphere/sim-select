import hydra

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg):
    print("hello world")



    

if __name__ == "__main__":
    main()