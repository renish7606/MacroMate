---
name: macromate-auth-fixes
description: >
  MacroMate authentication fixes skill. Use this when fixing two specific auth
  problems: (1) clicking "Try another image" or reset button logs the user out
  because session.flush() wipes the entire session including Django auth data,
  and (2) Google OAuth login skips account picker and logs in directly without
  showing which Google account to use. Both are one-line or two-line fixes.
  Apply this skill for any logout-on-reset or Google account-picker issues.
---

# MacroMate Auth Fixes Skill

## Two Problems, Two Fixes

---

## FIX 1 — "Try Another Image" Logs User Out

### Root Cause

In `detector/views.py`, the `reset_analysis` view calls:
```python
def reset_analysis(request):
    request.session.flush()       # THIS IS THE PROBLEM
    return redirect("index")
```

`session.flush()` deletes and recreates the entire session, including Django's
`_auth_user_id` key that keeps the user logged in. After flush, Django sees no
auth data, treats the user as anonymous, and redirects to login.

### Fix in `detector/views.py`

Replace the entire `reset_analysis` view:

```python
@login_required
def reset_analysis(request):
    """
    Clear only food-related session keys.
    Never touch auth session keys — that would log the user out.
    """
    FOOD_SESSION_KEYS = [
        "confirmed_food",
        "uploaded_image",
        "assistant_history",
        "manual_query",
    ]
    for key in FOOD_SESSION_KEYS:
        request.session.pop(key, None)   # safe delete — ignores missing keys

    request.session.modified = True
    return redirect("upload_food")       # back to upload page, still logged in
```

Make sure `login_required` is imported at the top of `views.py`:
```python
from django.contrib.auth.decorators import login_required
```

Also confirm the URL is still registered in `detector/urls.py`:
```python
path("reset/", reset_analysis, name="reset_analysis"),
```

---

## FIX 2 — Google OAuth Skips Account Picker

### Root Cause

By default Google OAuth reuses an existing browser Google session silently.
No `prompt` parameter is being sent so Google skips the account chooser and
logs in automatically with whatever account is already active in the browser.

### Fix A — `food_calorie_project/settings.py`

Add or update `SOCIALACCOUNT_PROVIDERS`:

```python
SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'SCOPE': ['profile', 'email'],
        'AUTH_PARAMS': {
            'access_type': 'online',
            'prompt': 'select_account',   # forces account picker every time
        },
        'OAUTH_PKCE_ENABLED': True,
    }
}
```

### Fix B — Create `detector/adapters.py` (most reliable method)

Create this new file:

```python
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter


class MacroMateSocialAccountAdapter(DefaultSocialAccountAdapter):
    """
    Forces Google account picker on every login attempt.
    Works at the adapter level — overrides any template or settings behavior.
    """
    def get_auth_params(self, request, action):
        params = super().get_auth_params(request, action)
        params["prompt"] = "select_account"
        return params
```

### Fix C — Register the adapter in `settings.py`

Add this line anywhere in `food_calorie_project/settings.py`:

```python
SOCIALACCOUNT_ADAPTER = 'detector.adapters.MacroMateSocialAccountAdapter'
```

### Fix D — Complete settings block to add/replace in `settings.py`

```python
# ── ALLAUTH ──
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

SITE_ID = 1

ACCOUNT_EMAIL_REQUIRED        = True
ACCOUNT_USERNAME_REQUIRED     = False
ACCOUNT_AUTHENTICATION_METHOD = 'email'
ACCOUNT_EMAIL_VERIFICATION    = 'none'

LOGIN_REDIRECT_URL  = '/'
LOGOUT_REDIRECT_URL = '/accounts/login/'

# ── GOOGLE OAUTH — forces account picker every time ──
SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'SCOPE': ['profile', 'email'],
        'AUTH_PARAMS': {
            'access_type': 'online',
            'prompt': 'select_account',
        },
        'OAUTH_PKCE_ENABLED': True,
    }
}

# Adapter enforces prompt at the Python level (most reliable)
SOCIALACCOUNT_ADAPTER = 'detector.adapters.MacroMateSocialAccountAdapter'

# ── SESSION — prevents random logout issues ──
SESSION_COOKIE_AGE          = 86400 * 7    # stay logged in 7 days
SESSION_SAVE_EVERY_REQUEST  = True          # refresh session on every request
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
```

---

## Verification Checklist

```
Fix 1 — Reset without logout:
[ ] Upload a food image → result page appears
[ ] Click "Try another image" → goes to /upload/ page (NOT /accounts/login/)
[ ] Username still shows in the navbar — still logged in
[ ] Upload another food → works fine

Fix 2 — Google account picker:
[ ] Log out of MacroMate
[ ] Click "Sign in with Google" → Google account picker appears
[ ] Can choose which account to use
[ ] Log out and sign in again → picker appears again, not silent login
[ ] Test in Incognito window for clearest result
```

---

## Decision Tree

```
"Try another image" still logs out after fix?
├── Confirm reset_analysis uses session.pop() not session.flush()
├── Confirm @login_required decorator is on the view
└── Confirm redirect goes to "upload_food" not "index"

Google still skips account picker?
├── Confirm settings.py has prompt: select_account in AUTH_PARAMS
├── Confirm detector/adapters.py file exists
├── Confirm SOCIALACCOUNT_ADAPTER is set in settings.py
└── Test in Incognito — Chrome caches Google sessions aggressively
    (may appear to skip picker if only 1 Google account exists in browser)

User gets logged out on browser refresh?
└── Add SESSION_SAVE_EVERY_REQUEST = True to settings.py
```

---

## Important Notes

1. `session.flush()` is the root cause of Fix 1. It is a nuclear wipe.
   Always use `session.pop(key, None)` to remove only specific keys.

2. `prompt=select_account` shows the account chooser.
   `prompt=consent` shows the full permissions screen every time.
   MacroMate only needs `select_account`.

3. The adapter in Fix B overrides the settings in Fix A if both exist.
   Apply both for maximum reliability — they reinforce each other.

4. `@login_required` on `reset_analysis` prevents unauthenticated users
   from hitting that view, which is a second category of logout bug.

5. The two fixes are completely independent — apply either one alone
   without breaking the other.
