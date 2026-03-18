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
