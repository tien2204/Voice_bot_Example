import asyncio
from pydantic import EmailStr
from mailjet_rest import Client
from app.core.config import settings


async def send_verification_email(
    email_to: EmailStr, username: str, verification_link: str
):
    """
    Sends a verification email to the user using Mailjet.
    Assumes MAILJET_API_KEY, MAILJET_API_SECRET, EMAILS_FROM_EMAIL, and EMAILS_FROM_NAME
    are configured in `app.core.config.settings`.
    """
    subject = "Verify your email address"
    body = f"Hi {username},\n\nPlease verify your email address by clicking the link below:\n{verification_link}\n\nThis link will expire in {settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES / 60:.0f} hours."

    logo_url = (
        f"{settings.NEXT_PUBLIC_APP_URL}/favicon.png"
        if hasattr(settings, "NEXT_PUBLIC_APP_URL") and settings.NEXT_PUBLIC_APP_URL
        else "your_logo_url_here/favicon.png"
    )
    app_name = (
        settings.NEXT_PUBLIC_APP_NAME
        if hasattr(settings, "NEXT_PUBLIC_APP_NAME") and settings.NEXT_PUBLIC_APP_NAME
        else "Your App"
    )
    html_body = f"""
<body style="margin: 0; padding: 0; background-color: 
    <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background-color: 
        <tr>
            <td align="center" style="padding: 20px 0;">
                <table width="600" border="0" cellspacing="0" cellpadding="0" style="max-width: 600px; margin: 0 auto; background-color: 
                    <tr>
                        <td align="center" style="padding: 24px 32px;">
                            <!-- Logo -->
                            <img src="{logo_url}" alt="{app_name} Logo" width="64" height="64" style="display: block; margin: 0 auto 24px auto; border-radius: 8px;">
                            <h1 style="margin-top: 0; margin-bottom: 24px; font-size: 24px; font-weight: 800; color: 
                                Verify your email for <span style="color: 
                            </h1>
                            <p style="margin-top: 0; margin-bottom: 16px; font-size: 16px; line-height: 1.5; color: 
                                Hi {username},
                            </p>
                            <p style="margin-top: 0; margin-bottom: 24px; font-size: 16px; line-height: 1.5; color: 
                                Please verify your email address by clicking the button below:
                            </p>
                            <!-- Verification Button -->
                            <table border="0" cellspacing="0" cellpadding="0" style="margin-bottom: 24px;">
                                <tr>
                                    <td align="center" bgcolor="
                                        <a href="{verification_link}" target="_blank" style="font-size: 14px; font-weight: 500; color: 
                                    </td>
                                </tr>
                            </table>
                            <p style="margin-top: 0; margin-bottom: 16px; font-size: 14px; line-height: 1.5; color: 
                                This link will expire in {settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES / 60:.0f} hours.
                            </p>
                            <p style="margin-top: 0; margin-bottom: 0; font-size: 14px; line-height: 1.5; color: 
                                If you did not request this email, please ignore it.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
"""

    required_settings_attrs = [
        "MAILJET_API_KEY",
        "MAILJET_API_SECRET",
        "EMAILS_FROM_EMAIL",
        "EMAILS_FROM_NAME",
    ]
    missing_settings = [
        s_attr
        for s_attr in required_settings_attrs
        if not (hasattr(settings, s_attr) and getattr(settings, s_attr))
    ]
    if missing_settings:
        print(
            f"Mailjet settings {', '.join(missing_settings)} are not configured or are empty. Skipping email sending."
        )

        return
    mailjet = Client(
        auth=(settings.MAILJET_API_KEY, settings.MAILJET_API_SECRET), version="v3.1"
    )
    message_data = {
        "Messages": [
            {
                "From": {
                    "Email": settings.EMAILS_FROM_EMAIL,
                    "Name": settings.EMAILS_FROM_NAME,
                },
                "To": [{"Email": email_to, "Name": username}],
                "Subject": subject,
                "TextPart": body,
                "HTMLPart": html_body,
            }
        ]
    }
    print(message_data)

    def send_sync():

        return mailjet.send.create(data=message_data)

    try:
        result = await asyncio.to_thread(send_sync)
        if result.status_code == 200:
            print(
                f"Verification email successfully sent to {email_to} via Mailjet. Respond: {result.json()}"
            )
        else:
            print(
                f"Failed to send verification email to {email_to} via Mailjet. Status: {result.status_code}, Response: {result.json()}"
            )

    except Exception as e:
        print(f"An error occurred while sending email via Mailjet: {e}")
