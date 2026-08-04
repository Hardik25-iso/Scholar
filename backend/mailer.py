"""Outbound email over SMTP — stdlib only, no new dependency.

Two things in this app are useless without mail: a forgotten password (the
account is locked out permanently) and a workspace invitation (the recipient
never learns the code). Both previously logged their token, which is fine on a
laptop and an account-takeover vector on a server.

Three rules this module exists to enforce:

1. **Sending never raises.** `send` returns True/False. A mail server that is
   down must not turn `/auth/forgot` into a 500 — that would also make the
   route an account-existence oracle, since a nonexistent account skips
   delivery entirely and would keep answering 202.
2. **Unconfigured is a state, not an error.** With no SMTP host set the app
   still runs; callers fall back to logging and say so.
3. **The password never reaches a log.** Only the host, port and recipient are
   ever logged.
"""
import logging
import smtplib
import ssl
from email.message import EmailMessage

from backend.config import settings

log = logging.getLogger(__name__)


def configured() -> bool:
    """Whether real delivery is possible. Callers branch on this to stay honest
    about whether a message was actually sent."""
    return bool(settings.smtp_host and settings.mail_from)


def send(to: str, subject: str, body: str) -> bool:
    """Deliver one plain-text message. Returns whether it was accepted.

    Plain text rather than HTML on purpose: these messages carry a link and
    nothing else, and a text/plain body cannot be used to disguise where that
    link points.
    """
    if not configured():
        return False

    message = EmailMessage()
    message["From"] = settings.mail_from
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    try:
        _transmit(message)
    except (OSError, smtplib.SMTPException) as exc:
        # The exception text can name the host and the recipient; it never
        # contains the password, which smtplib only ever puts on the wire.
        log.error("mail to %s failed via %s:%s — %s",
                  to, settings.smtp_host, settings.smtp_port, exc)
        return False

    log.info("mail sent to %s via %s:%s", to, settings.smtp_host, settings.smtp_port)
    return True


def _transmit(message: EmailMessage) -> None:
    """Open a connection, authenticate if credentials exist, and send.

    Both encrypted modes are supported because providers differ: implicit TLS on
    port 465 (SMTP_SSL) and opportunistic STARTTLS on 587. `smtp_starttls` may
    be turned off only for a local relay — over the public internet it would put
    the SMTP password on the wire in clear text.
    """
    context = ssl.create_default_context()
    factory = smtplib.SMTP_SSL if settings.smtp_ssl else smtplib.SMTP
    kwargs = {"context": context} if settings.smtp_ssl else {}

    with factory(settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout,
                 **kwargs) as server:
        if settings.smtp_starttls and not settings.smtp_ssl:
            server.starttls(context=context)
        if settings.smtp_user:
            server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(message)


def link(path: str, token: str) -> str:
    """Build a user-facing link into the frontend.

    Uses `public_app_url` rather than `frontend_origin`: the origin the browser
    is allowed to call the API from and the address a link in an email should
    point at are the same thing in development and often not in production
    (a proxy, a custom domain). Conflating them would put `localhost` in real
    email the first time this is deployed.
    """
    base = (settings.public_app_url or settings.frontend_origin).rstrip("/")
    return f"{base}{path}?token={token}"
