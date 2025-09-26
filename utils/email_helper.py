import smtplib, os
from email.mime.text import MIMEText

def send_email(subject: str, body: str):
    smtp_host = "pro3.mail.ovh.net"  # OVH
    smtp_port = 587
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_addr   = os.getenv("NOTIFY_EMAIL", smtp_user)

    if not (smtp_user and smtp_pass and to_addr):
        return  # silently skip if not configured

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)
