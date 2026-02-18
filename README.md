\# Product Classification Project



Ovaj projekat implementira model mašinskog učenja za klasifikaciju proizvoda na osnovu njihovog naziva (Product Title).

Model koristi TF-IDF vektorizaciju teksta i Logistic Regression algoritam za klasifikaciju u odgovarajuće kategorije.





\## Tehnologije

* Python
* Pandas
* Scikit-learn
* TF-IDF (TfidfVectorizer)
* Logistic Regression
* Joblib





\## Sadržaj repozitorijuma

* product\_classification\_analysis.ipynb - analiza podataka i razvoj modela
* train\_model.py - skripta za treniranje modela
* predict\_category.py - skripta za predikciju kategorije
* product\_classifier.pkl - sačuvani trenirani model
* products.csv - dataset korišten u projektu





\## Kako pokrenuti projekat


1. Instalirati potrebne biblioteke:

pip install pandas scikit-learn joblib



2\. Pokrenuti treniranje modela:

python train\_model.py



3\. Pokrenuti predikciju:

python predict\_category.py



Nakon pokretanja skripte za predikciju, potrebno je unijeti naziv proizvoda, a model će ispisati predviđenu kategoriju.

