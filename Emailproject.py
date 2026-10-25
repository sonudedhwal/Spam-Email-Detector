"""
SpamShield  -  Email / SMS spam detector     Python + ML (scikit-learn) + Streamlit + MySQL (+ optional LLM)
=========================================================================================================
Run locally :  pip install -r requirements.txt  &&  streamlit run Emailproject.py

DEPLOYING ON STREAMLIT CLOUD  (fixes "ModuleNotFoundError: No module named 'sklearn'")
    1. requirements.txt must sit in the ROOT of your GitHub repo (next to this file) and be named exactly that.
    2. It must say  scikit-learn  (NOT "sklearn").  Use the requirements.txt that comes with this project.
    3. Commit + push, then  Manage app -> Reboot app  (or delete and redeploy). Python 3.11 or 3.12 is fine.
    4. MySQL: Streamlit Cloud cannot reach localhost / XAMPP. Use a hosted MySQL (Aiven, TiDB Cloud, Railway, ...) and put
       DATABASE_URL in  Manage app -> Settings -> Secrets. Without it the app falls back to a SQLite file
       (works, but is wiped whenever the cloud app restarts).

What it does
    Detect    one email: spam probability, risk signals, the words that pushed the verdict (model-agnostic), optional LLM
              explanation, and a feedback button that teaches the model.
    Bulk      classify a whole CSV and download the result.
    Train     compare Naive Bayes / Logistic Regression / Linear SVM / Random Forest (TF-IDF or CountVectorizer) with a
              hold-out test + cross-validation, inspect the confusion matrix and top words, activate the best model.
    History   MySQL-backed log, spam-rate chart, retrain from your feedback, delete all stored data.
"""
import io
import json
import os
import re
import urllib.request
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="SpamShield", page_icon="🛡️", layout="wide")

try:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sqlalchemy import (Column, DateTime, Float, Integer, LargeBinary, MetaData, String, Table, Text, create_engine,
                            delete, insert, select, text)
    from sqlalchemy.dialects.mysql import LONGBLOB
    from sqlalchemy.engine import make_url
except ModuleNotFoundError as _e:        # the exact error you saw: a package is missing on the server
    st.error(f"Missing Python package: **{_e.name}**. The server did not install your requirements.")
    st.markdown("**Fix:** add a file named `requirements.txt` to the **root of your GitHub repo** with this content, push, "
                "then **Manage app → Reboot app**:")
    st.code("streamlit>=1.36\npandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\njoblib>=1.3\nsqlalchemy>=2.0\npymysql>=1.1\ncryptography>=42",
            language="text")
    st.caption("Write `scikit-learn`, not `sklearn`. Do not list built-in modules such as pickle, json or sqlite3.")
    st.stop()

SEED = 42
LLM_MODEL = os.environ.get("SPAM_LLM_MODEL", "claude-sonnet-5-5")


# =====================================================================
# 1. BUILT-IN SAMPLE DATA (so the app works the moment it is deployed; upload a bigger CSV for real accuracy)
# =====================================================================
_SPAM = [
    "WINNER!! You have won a $1000 Walmart gift card. Click http://bit.ly/claim-now to claim your prize today",
    "Congratulations! You've been selected for a FREE iPhone. Reply YES now to claim. Limited time offer!!!",
    "URGENT: Your bank account has been suspended. Verify your password immediately at http://secure-login.xyz",
    "Get cheap viagra and cialis online. No prescription needed. Order now and save 80%",
    "You have won the lottery of 5,000,000 dollars. Send your bank details to receive the funds",
    "FREE entry in our weekly competition to win tickets. Text WIN to 80085 now. T&Cs apply",
    "Dear customer, your PayPal account is limited. Confirm your identity here to avoid permanent closure",
    "Make $5000 a week working from home! No experience needed. Click here to start earning today",
    "Hot singles in your area are waiting to meet you. Click here for free registration",
    "Your package could not be delivered. Pay a small fee of $1.99 at http://post-office-track.top to reschedule",
    "Earn money fast with this one weird trick. Millions already joined. Act now before it is gone!",
    "Final notice: your car warranty is about to expire. Call 1-800-555-0199 now to renew and save big",
    "CONGRATULATIONS you are our lucky winner of a cash prize of 10000. Call 09061701461 to claim",
    "Increase your income with crypto trading. Guaranteed 300% profit in 7 days. Invest now!!!",
    "Buy followers and likes cheap. 100% real. Discount code SAVE50 valid today only",
    "Your Netflix payment failed. Update your billing information immediately to keep your account active",
    "Exclusive deal just for you: 90% off luxury watches. Free shipping worldwide. Order today",
    "You are pre-approved for a loan of up to 50000. No credit check. Apply in 2 minutes, click the link",
    "Claim your free vacation to the Bahamas. Just pay the processing fee. Offer expires tonight!",
    "Dear friend, I am a prince and need your help to transfer 25 million dollars. You will get 30 percent",
    "Lose 20 pounds in 2 weeks with this miracle pill. Doctors hate it. Order your free trial now",
    "ALERT: unusual sign-in detected on your account. Click to verify now or your account will be deleted",
    "Free ringtones and wallpapers! Text TONE to 87021 to subscribe. 3 pounds per week",
    "Work from home and get paid daily. Send your name and phone number to join our team right now",
    "Win a brand new car! Enter our giveaway today. No purchase necessary. Click to enter",
    "Your invoice is attached. Open the attachment to view payment details immediately",
    "Meet rich singles tonight. Join free and chat now. Over 18s only. Click the link below",
    "Cheap meds online, best prices, fast delivery, no doctor visit. Buy two get one free",
    "Act now! Your tax refund of 1,847.20 is waiting. Submit your card details to receive it",
    "SIX chances to win CASH! From 100 to 20,000 pounds. Txt CSH11 to 87575. Cost 150p per day",
]
_HAM = [
    "Hi team, the project meeting is moved to 3 pm tomorrow in the second floor conference room.",
    "Can you send me the slides from yesterday's presentation? I want to review them before the client call.",
    "Mom called, she wants us to come for dinner on Sunday. Let me know if that works for you.",
    "Reminder: your dentist appointment is on Friday at 10:30. Please arrive ten minutes early.",
    "Thanks for the update. I have reviewed the report and added a few comments in the shared document.",
    "Happy birthday! Hope you have a wonderful day. Let's catch up over coffee this weekend.",
    "The invoice for October is attached for your records. Please let me know if anything looks off.",
    "I'll be late today, the train is delayed. Please start the stand-up without me.",
    "Could you please confirm the delivery address for the order we discussed on Monday?",
    "Lunch at the usual place at 1? I'll book a table for four people.",
    "Attached is the revised proposal. Let me know if you have any questions before we send it out.",
    "Don't forget to submit your timesheet by Friday evening so payroll can be processed on time.",
    "Great job on the presentation today, the client was really impressed with the numbers.",
    "I found your keys on the kitchen table. I'll leave them with the neighbour if I leave early.",
    "The server maintenance is scheduled for Saturday night. Expect short downtime between 11 and midnight.",
    "Are we still on for the football match tonight? I can pick you up around seven.",
    "Please find the minutes of the last meeting below. Action items are marked in bold.",
    "Hey, just checking in. How did the interview go? Call me when you are free.",
    "Your library book is due next week. You can renew it online or at the front desk.",
    "I booked the flight tickets for the conference. The itinerary is attached to this email.",
    "Can we reschedule our call to Thursday morning? I have a clash with another meeting.",
    "The kids' school trip is on the twelfth. I signed the permission slip and left it in your bag.",
    "Thanks for the recommendation, I read the book and really enjoyed it. Any others like it?",
    "Quarterly sales figures are ready. I will present the highlights in tomorrow's review.",
    "Sorry I missed your call. I was in a meeting. I'll ring you back in half an hour.",
    "Your order has been shipped and should arrive on Wednesday. Tracking details are in your account.",
    "Let's go over the budget draft on Monday morning. I have a few questions about the travel line.",
    "Could you water the plants while I am away next week? The spare key is under the mat.",
    "The code review is done. I left comments on two functions, otherwise it looks good to merge.",
    "Dinner was lovely, thank you for hosting. We should do it again at our place next month.",
]
SAMPLE = pd.DataFrame({"text": _SPAM + _HAM, "label": [1] * len(_SPAM) + [0] * len(_HAM)})


# =====================================================================
# 2. DATABASE  (MySQL / MariaDB via SQLAlchemy; automatic SQLite fallback so the app never crashes)
# =====================================================================
def _db_url():
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        try:
            url = str(st.secrets.get("DATABASE_URL", ""))
        except Exception:
            url = ""
    return url.strip() or "mysql+pymysql://root:@localhost:3306/spamshield"          # XAMPP default


meta = MetaData()
predictions = Table("predictions", meta,
                    Column("id", Integer, primary_key=True, autoincrement=True), Column("created_at", DateTime),
                    Column("text", Text), Column("label", String(4)), Column("spam_prob", Float),
                    Column("model", String(80)), Column("source", String(12)))
training_data = Table("training_data", meta,
                      Column("id", Integer, primary_key=True, autoincrement=True), Column("created_at", DateTime),
                      Column("text", Text), Column("label", Integer), Column("source", String(20)),
                      Column("prediction_id", Integer))
models = Table("models", meta,
               Column("id", Integer, primary_key=True, autoincrement=True), Column("created_at", DateTime),
               Column("name", String(80)), Column("sklearn", String(20)), Column("metrics", Text), Column("n_train", Integer),
               Column("blob", LargeBinary().with_variant(LONGBLOB(), "mysql", "mariadb")))


def _make_engine(url):
    u = make_url(url)
    kw = {"pool_pre_ping": True, "pool_recycle": 1800}
    if u.get_backend_name() in ("mysql", "mariadb"):
        kw["connect_args"] = {"connect_timeout": 4}
        if u.database:
            if not re.fullmatch(r"[A-Za-z0-9_]+", u.database):
                raise ValueError("The database name may only contain letters, digits and underscores.")
            srv = create_engine(u.set(database=None), **kw)
            with srv.begin() as c:
                c.execute(text(f"CREATE DATABASE IF NOT EXISTS `{u.database}` CHARACTER SET utf8mb4"))
            srv.dispose()
    eng = create_engine(u, **kw)
    meta.create_all(eng)
    return eng


@st.cache_resource(show_spinner="Connecting to the database...")
def db():
    """-> (engine, backend name, reason MySQL was not used). Never raises."""
    url = _db_url()
    try:
        return _make_engine(url), make_url(url).get_backend_name(), ""
    except Exception as e:
        reason = f"{type(e).__name__}: {str(e)[:160]}"
        return _make_engine("sqlite:///spamshield.db"), "sqlite", reason


def now():
    return datetime.now().replace(microsecond=0)


def save_prediction(txt, label, prob, model, source="single"):
    with db()[0].begin() as c:
        return c.execute(insert(predictions).values(created_at=now(), text=txt[:1500], label=label, spam_prob=float(prob),
                                                    model=model[:80], source=source)).inserted_primary_key[0]


def add_training(txt, label, source, prediction_id=None):
    with db()[0].begin() as c:
        c.execute(insert(training_data).values(created_at=now(), text=txt[:5000], label=int(label), source=source,
                                               prediction_id=prediction_id))


def add_training_df(df, source):
    rows = [dict(created_at=now(), text=t[:5000], label=int(l), source=source, prediction_id=None) for t, l in zip(df["text"], df["label"])]
    with db()[0].begin() as c:
        if rows:
            c.execute(insert(training_data), rows)
    return len(rows)


def read_sql(stmt):
    with db()[0].connect() as c:
        return pd.read_sql(stmt, c)


def saved_training():
    return read_sql(select(training_data.c.text, training_data.c.label, training_data.c.source))


# =====================================================================
# 3. MACHINE LEARNING
# =====================================================================
MODEL_CHOICES = ["Multinomial Naive Bayes", "Complement Naive Bayes", "Logistic Regression", "Linear SVM (calibrated)", "Random Forest"]
VECTORIZERS = ["TF-IDF (unigrams + bigrams)", "Count words (CountVectorizer)"]


def build_pipe(model_name, vec_name, y=None):
    if vec_name.startswith("TF-IDF"):
        vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    else:
        vec = CountVectorizer(lowercase=True, strip_accents="unicode", stop_words="english", ngram_range=(1, 2))
    if model_name == "Multinomial Naive Bayes":
        clf = MultinomialNB(alpha=0.3)
    elif model_name == "Complement Naive Bayes":
        clf = ComplementNB(alpha=0.3)
    elif model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "Linear SVM (calibrated)":
        min_count = int(pd.Series(y).value_counts().min()) if y is not None else 3
        clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=max(2, min(3, min_count)))
    else:
        clf = RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    return Pipeline([("vec", vec), ("clf", clf)])


def spam_prob(pipe, texts):
    clf = pipe.named_steps["clf"]
    X = pipe.named_steps["vec"].transform(list(texts))
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    return 1 / (1 + np.exp(-clf.decision_function(X)))


def evaluate(df, model_names, vec_name, test_size):
    y = df["label"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], y, test_size=test_size, stratify=y, random_state=SEED)
    k = min(5, int(pd.Series(y).value_counts().min()))
    out = []
    for name in model_names:
        pipe = build_pipe(name, vec_name, y_tr).fit(X_tr, y_tr)
        prob = spam_prob(pipe, X_te)
        pred = (prob >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary", pos_label=1, zero_division=0)
        try:
            auc = float(roc_auc_score(y_te, prob))
        except ValueError:
            auc = float("nan")
        cv_f1 = float("nan")
        if k >= 2:
            try:
                cv_f1 = float(cross_val_score(build_pipe(name, vec_name, y), df["text"], y, scoring="f1",
                                              cv=StratifiedKFold(k, shuffle=True, random_state=SEED)).mean())
            except Exception:
                pass
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        out.append({"model": name, "pipe": pipe, "accuracy": accuracy_score(y_te, pred), "precision": p, "recall": r, "f1": f,
                    "auc": auc, "cv_f1": cv_f1, "cm": cm, "n_test": len(y_te)})
    return sorted(out, key=lambda d: (d["f1"], d["auc"] if d["auc"] == d["auc"] else 0), reverse=True)


def top_words(pipe, n=15):
    """-> (spam words DataFrame, ham words DataFrame) with a signed weight, or None when the model has no weights."""
    names = np.array(pipe.named_steps["vec"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        w = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif hasattr(clf, "calibrated_classifiers_"):
        w = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    else:
        return None
    order = np.argsort(w)
    return (pd.DataFrame({"word": names[order[::-1][:n]], "weight": w[order[::-1][:n]]}),
            pd.DataFrame({"word": names[order[:n]], "weight": w[order[:n]]}))


def explain(pipe, txt, topn=8):
    """Model-agnostic: how much does the spam probability drop when each word is removed? (positive = pushes towards spam)"""
    base = float(spam_prob(pipe, [txt])[0])
    words = list(dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z']{3,}", txt)))[:50]
    if not words:
        return pd.DataFrame(columns=["word", "effect"])
    variants = [re.sub(rf"\b{re.escape(w)}\b", " ", txt, flags=re.I) for w in words]
    probs = spam_prob(pipe, variants)
    df = pd.DataFrame({"word": words, "effect": base - probs})
    df = df[df["effect"].abs() > 1e-4].sort_values("effect", key=abs, ascending=False).head(topn)
    return df.sort_values("effect")


def pack(pipe, name, metrics, n):
    return {"pipe": pipe, "name": name, "metrics": metrics, "n": n, "trained_at": now(), "sklearn": sklearn.__version__}


def persist_model(bundle):
    buf = io.BytesIO()
    joblib.dump(bundle["pipe"], buf, compress=3)
    with db()[0].begin() as c:
        c.execute(insert(models).values(created_at=now(), name=bundle["name"], sklearn=sklearn.__version__,
                                        metrics=json.dumps(bundle["metrics"]), n_train=bundle["n"], blob=buf.getvalue()))
        keep = [r[0] for r in c.execute(select(models.c.id).order_by(models.c.id.desc()).limit(5))]
        c.execute(delete(models).where(models.c.id.not_in(keep)))


def load_persisted():
    try:
        with db()[0].connect() as c:
            r = c.execute(select(models).order_by(models.c.id.desc()).limit(1)).mappings().first()
        if r and r["sklearn"] == sklearn.__version__:
            b = pack(joblib.load(io.BytesIO(r["blob"])), r["name"], json.loads(r["metrics"] or "{}"), r["n_train"])
            b["trained_at"] = r["created_at"]
            return b
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner="Preparing the spam model...")
def holder():
    h = {"bundle": load_persisted()}
    if h["bundle"] is None:
        y = SAMPLE["label"].to_numpy()
        pipe = build_pipe("Logistic Regression", VECTORIZERS[0], y).fit(SAMPLE["text"], y)
        h["bundle"] = pack(pipe, "Logistic Regression (built-in sample data)", {}, len(SAMPLE))
        try:
            persist_model(h["bundle"])
        except Exception:
            pass
    return h


def bundle():
    return holder()["bundle"]


def activate(model_name, vec_name, df, metrics):
    """Train on ALL given rows and make it the live model."""
    y = df["label"].to_numpy()
    pipe = build_pipe(model_name, vec_name, y).fit(df["text"], y)
    b = pack(pipe, f"{model_name} · {vec_name.split(' ')[0]}", metrics, len(df))
    holder()["bundle"] = b
    persist_model(b)
    return b


# ---- dataset helpers ----
def to_label(v):
    s = str(v).strip().lower()
    if s in ("spam", "1", "1.0", "true", "yes", "junk", "phishing"):
        return 1
    if s in ("ham", "0", "0.0", "false", "no", "not spam", "legitimate", "normal"):
        return 0
    return np.nan


def read_dataset(uploaded):
    raw, df = uploaded.getvalue(), None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, index_col=False)
            break
        except Exception:
            continue
    if df is None or df.empty:
        raise ValueError("That file could not be read as a CSV.")
    cols = {str(c).lower().strip(): c for c in df.columns}
    tcol = next((cols[k] for k in ("text", "message", "v2", "email", "body", "content", "mail") if k in cols), None)
    lcol = next((cols[k] for k in ("label", "v1", "category", "class", "spam", "target", "type") if k in cols), None)
    if tcol is None or lcol is None:
        if df.shape[1] < 2:
            raise ValueError("The CSV needs a text column and a label column (spam / ham).")
        lcol, tcol = df.columns[0], df.columns[1]
    out = pd.DataFrame({"text": df[tcol].astype(str).str.strip(), "label": df[lcol].map(to_label)}).dropna()
    out = out[out["text"].str.len() > 0].drop_duplicates("text")
    out["label"] = out["label"].astype(int)
    if out["label"].nunique() < 2 or out["label"].value_counts().min() < 3:
        raise ValueError("Need at least 3 spam and 3 non-spam messages with recognisable labels (spam/ham, 1/0).")
    return out.reset_index(drop=True)


# ---- rule-based signals (explainable, works without any model) ----
URGENT = ["urgent", "immediately", "act now", "final notice", "expires", "limited time", "verify", "suspended", "alert", "last chance"]
MONEY = ["free", "winner", "won", "prize", "cash", "lottery", "congratulations", "guaranteed", "discount", "% off", "100%", "earn", "profit",
         "loan", "credit", "refund", "gift card", "bitcoin", "crypto"]
CRED = ["password", "bank details", "verify your", "confirm your identity", "login", "card details", "otp", "pin"]


def signals(txt):
    t = txt.lower()
    out = []
    urls = re.findall(r"https?://\S+|www\.\S+", t)
    if urls:
        out.append((f"{len(urls)} link(s)" + (" using a URL shortener" if any(s in u for u in urls for s in ("bit.ly", "tinyurl", "goo.gl", "t.co")) else ""), "warn"))
    if any(re.search(r"https?://\S+\.(xyz|top|click|loan|work|gq|tk)\b", u) for u in urls):
        out.append(("link with a suspicious domain ending", "bad"))
    for label, words, tone in (("urgency words", URGENT, "warn"), ("money / prize words", MONEY, "warn"), ("asks for credentials or card data", CRED, "bad")):
        hit = [w for w in words if w in t]
        if hit:
            out.append((f"{label}: {', '.join(hit[:5])}", tone))
    letters = [ch for ch in txt if ch.isalpha()]
    if len(letters) > 20 and sum(ch.isupper() for ch in letters) / len(letters) > 0.35:
        out.append(("a lot of CAPITAL LETTERS", "warn"))
    if txt.count("!") >= 3:
        out.append((f"{txt.count('!')} exclamation marks", "warn"))
    if re.search(r"(\+?\d[\d\s\-]{8,}\d)", txt):
        out.
