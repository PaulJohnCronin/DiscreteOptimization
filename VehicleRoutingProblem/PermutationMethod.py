def permutations(s):
    if len(s) <= 1:
        return [s]
    else:
        perms = []
        for e in permutations(s[:-1]):
            for i in range(len(e)+1):
                perms.append(e[:i] + s[-1] + e[i:])
                print([e[:i] + s[-1] + e[i:]])
        return perms

permutations('012345')