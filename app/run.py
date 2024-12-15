import os
import subprocess 

def run_commands():
    os.chdir("..")
    
    print("Chạy docker-compose trong thư mục qdrant...")
    os.chdir("qdrant")
    subprocess.run(["docker-compose", "up", "-d"])

    os.chdir("..")
    
    print("Chạy server bằng uvicorn...") 
    subprocess.run(["uvicorn", "app.main:app", "--reload"])

if __name__ == "__main__":
    run_commands()
