import os
import shutil
import datetime
import argparse
from src.utils.logger import LOGS_DIR, setup_daily_log_rotation

def archive_logs(days_to_keep=30):
    """Archive logs older than the specified number of days"""
    daily_dir = os.path.join(LOGS_DIR, "daily")
    if not os.path.exists(daily_dir):
        return
        
    # Get current date
    current_date = datetime.datetime.now()
    
    # Check each daily log directory
    for date_dir in os.listdir(daily_dir):
        try:
            # Parse the date from directory name
            dir_date = datetime.datetime.strptime(date_dir, "%Y-%m-%d")
            
            # Calculate age in days
            age_days = (current_date - dir_date).days
            
            # If older than threshold, archive or delete
            if age_days > days_to_keep:
                dir_path = os.path.join(daily_dir, date_dir)
                
                # Option 1: Delete old logs
                # shutil.rmtree(dir_path)
                # print(f"Deleted old logs from {date_dir}")
                
                # Option 2: Archive old logs
                archive_dir = os.path.join(LOGS_DIR, "archive")
                os.makedirs(archive_dir, exist_ok=True)
                
                # Create archive file name
                archive_file = os.path.join(archive_dir, f"logs_{date_dir}.zip")
                
                # Create zip archive
                shutil.make_archive(
                    os.path.splitext(archive_file)[0],  # Remove .zip extension
                    'zip',
                    dir_path
                )
                
                # Remove the original directory
                shutil.rmtree(dir_path)
                print(f"Archived logs from {date_dir}")
        except Exception as e:
            print(f"Error processing log directory {date_dir}: {str(e)}")

def clear_logs():
    """Clear all logs (for development/testing)"""
    for filename in os.listdir(LOGS_DIR):
        file_path = os.path.join(LOGS_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")
    print("All logs cleared")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage application logs")
    parser.add_argument("--rotate", action="store_true", help="Rotate logs to daily directory")
    parser.add_argument("--archive", type=int, default=30, help="Archive logs older than N days")
    parser.add_argument("--clear", action="store_true", help="Clear all logs (use with caution)")
    
    args = parser.parse_args()
    
    if args.rotate:
        setup_daily_log_rotation()
        print("Logs rotated to daily directory")
        
    if args.archive:
        archive_logs(args.archive)
        print(f"Logs older than {args.archive} days archived")
        
    if args.clear:
        confirm = input("Are you sure you want to clear all logs? (y/n): ")
        if confirm.lower() == 'y':
            clear_logs()
            print("All logs cleared")