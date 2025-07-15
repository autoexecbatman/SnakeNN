@echo off
echo Initializing git repository and connecting to GitHub...
echo.

cd /d "D:\repo\snakeNN"

echo Step 1: Initialize git repository
git init
echo.

echo Step 2: Add GitHub remote repository
git remote add origin https://github.com/autoexecbatman/SnakeNN.git
echo.

echo Step 3: Check remote connection
git remote -v
echo.

echo Step 4: Add all files to staging
git add .
echo.

echo Step 5: Create initial commit
git commit -m "Initial commit: Complete SnakeNN project with DQN implementation"
echo.

echo Step 6: Set main branch and push to GitHub
git branch -M main
echo.

echo Step 7: Push to GitHub (this will populate your empty GitHub repo)
git push -u origin main
echo.

echo Repository successfully initialized and pushed to GitHub!
echo.
pause
