from data import DataProcessor

def main():
    processor = DataProcessor()
    processor.load_data()
    processor.normalize_data()
    processor.split_data()
    return 

if __name__ == "__main__":
    main()
