To get started, first clone this repo:

```
git clone https://github.com/plotly/dash-sample-apps.git
cd dash-sample-apps/apps/dash-app
```

where `dash-app` is the app you would like to access.


Create and activate a conda env:
```
conda create -n dash-venv python=3.7.6
conda activate dash-venv
```

Or a venv (make sure your `python3` is 3.6+):
```
python3 -m venv venv
source venv/bin/activate  # for Windows, use venv\Scripts\activate.bat
```

Install all the requirements:

```
pip install -r requirements.txt
```

You can now run the app:
```
python app.py
```

and visit http://127.0.0.1:8050/.


### Windows virtualenv

If you are on Windows, and you cannot use Windows Subsystem for Linux (WSL).
