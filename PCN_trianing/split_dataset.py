"""
This is a helper code used to split the Kaggle dataset into a Train, Test, Validation split to ue for experiments later.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

def split_dataset(csv_path, original_img_dir, output_base_dir):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    train_img_dir = os.path.join(output_base_dir, 'train')
    test_img_dir = os.path.join(output_base_dir, 'test')
    val_img_dir = os.path.join(output_base_dir, 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # Get unique classes
    classes = df['label'].unique()

    # Initialize dataframes for splits
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)

    # Stratified split
    for cls in classes:
        # Get images for this class
        class_df = df[df['label'] == cls]

        # First split: 80% train, 20% (test + val)
        train_temp, test_val_temp = train_test_split(class_df, test_size=0.2, random_state=42)

        # Split the remaining 20% into test and validation
        test_temp, val_temp = train_test_split(test_val_temp, test_size=0.5, random_state=42)

        # Append to respective dataframes
        train_df = pd.concat([train_df, train_temp])
        test_df = pd.concat([test_df, test_temp])
        val_df = pd.concat([val_df, val_temp])

    # Shuffle the dataframes
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Copy images to respective directories and create new CSVs
    def copy_images(df, dest_dir, dest_csv_path):
        dest_df = df.copy()
        for index, row in df.iterrows():
            src_path = os.path.join(original_img_dir, row['filename'])
            dest_filename = row['filename']
            dest_path = os.path.join(dest_dir, dest_filename)
            shutil.copy2(src_path, dest_path)
            dest_df.loc[index, 'filename'] = dest_filename

        dest_df.to_csv(dest_csv_path, index=False)

    # Copy images and create split CSVs
    copy_images(train_df, train_img_dir, os.path.join(output_base_dir, 'train_labels.csv'))
    copy_images(test_df, test_img_dir, os.path.join(output_base_dir, 'test_labels.csv'))
    copy_images(val_df, val_img_dir, os.path.join(output_base_dir, 'val_labels.csv'))

    # Print split statistics
    print("Dataset Split Statistics:")
    print(f"Total Images: {len(df)}")
    print(f"Training Images: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Test Images: {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)")
    print(f"Validation Images: {len(val_df)} ({len(val_df)/len(df)*100:.2f}%)")

    print("\nClass Distribution:")
    for split, df_split in [("Training", train_df), ("Test", test_df), ("Validation", val_df)]:
        print(f"\n{split} Set:")
        print(df_split['label'].value_counts())

# Usage
split_dataset('har/data_labels.csv', 'har/data', 'har/splits')