def user_context(request):
    user = getattr(request, "user", None)
    role = "GUEST"
    user_initial = "U"

    if user and getattr(user, "is_authenticated", False):
        group_names = {group.name.upper() for group in user.groups.all()}
        if "ADMIN" in group_names:
            role = "ADMIN"
        elif "EDUCATOR" in group_names:
            role = "EDUCATOR"
        elif "STUDENT" in group_names:
            role = "STUDENT"
        elif user.is_staff:
            role = "ADMIN"
        user_initial = (user.username[:1] if user.username else "U").upper()

    return {
        "user_role": role,
        "user_role_display": role.title() if role != "GUEST" else "Guest",
        "user_initial": user_initial,
    }
