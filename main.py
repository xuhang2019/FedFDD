from server import fedavg, fedfdd
import argparse

def get_server(name='fedfdd'):
    if name.lower() == 'fedfdd':
        return fedfdd.FedFDDServer()
    elif name.lower() == 'fedavg':
        return fedavg.FedAvgServer()
    else:
        raise ValueError(f"Invalid server name: {name}")
    
def main():
    server = get_server(args.fed_name)
    server.run()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fed_name', type=str, default='fedfdd', help='Federated Learning Algorithm Name')
    args = parser.parse_args()
    main()