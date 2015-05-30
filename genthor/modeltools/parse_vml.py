
def parse(fn):
    lines = open(fn, 'rU').read().split('\n')
    i = 0
    N = len(lines)
    for i in range(N):
        l = lines[i]
        if 'material' in l:
            break
    appearance_lines = []
    for j in range(i+1, N):
        l = lines[j]
        if '}' in l:
            break
        appearance_lines.append(l)
    for i in range(j, N):
        l = lines[i]
        if 'coord' in l:
            break
    geom_lines = []
    for j in range(i+1, N):
        l = lines[j]
        if ']' in l:
            break
        geom_lines.append(l)
    for i in range(j, N):
        l = lines[i]
        if 'texCoord' in l:
            break
    texCoords = []
    for j in range(i+1, N):
        l = lines[j]
        if ']' in l:
            break
        texCoords.append(l)
    for i in range(j, N):
        l = lines[i]
        if 'texCoordIndex' in l:
            break
    texCoordIndex = []
    for j in range(i+1, N):
        l = lines[j]
        if ']' in l:
            break
        texCoordIndex.append(l)
    for i in range(j, N):
        l = lines[i]
        if 'coordIndex' in l:
            break
    coordIndex = []
    for j in range(i+1, N):
        l = lines[j]
        if ']' in l:
            break
        coordIndex.append(l)
            
    appearance_lines = parse_appearance(appearance_lines)
    geom_lines = parse_coords(geom_lines)
    texCoords = parse_coords(texCoords)
    texCoordIndex = parse_index(texCoordIndex)
    coordIndex = parse_index(coordIndex)
    return appearance_lines, geom_lines, texCoords, coordIndex, texCoordIndex
    
            
def parse_appearance(X):
    def parse_ap(l):
        l = l.strip(' \t')
        l = l.split(' ')
        return l[0], map(float, l[1:])
    X = map(parse_ap, X)
    return dict(X)

def parse_coords(X):
    def parse_c(l):
        l = l.strip(' \t,')
        return map(float, l.split(' '))
    return map(parse_c, X)

def parse_index(X):
    def parse_i(l):
        l = l.strip(' \t,').split(', ')
        return [1 + _x for _x in map(int, l[:-1])]
    return map(parse_i, X)


tdict = {'ambientIntensity': 'Ka', 
         'diffuseColor': 'Kd', 
         'shininess': 'Ns', 
         'specularColor': 'Ks',
         'transparency': 'd'}

    
def write_obj(fn, jpgfl, app, geom, texCoords, coordIndex, texCoordIndex):
    mtl_lines = ['newmtl Mat']
    for a in app:
        n = tdict[a]
        v = app[a]
        if n in ['Kd', 'Ka', 'Ks'] and len(v) == 1:
            v = 3 * v
        v = ' '.join(map(str, v))
        newl = n + ' ' + v
        mtl_lines.append(newl)
    mtl_lines.append('illum 2')
    mtl_lines.append('map_Kd %s\n' % os.path.split(jpgfl)[-1])

    mtl_fn = fn + '.mtl'
    with open(mtl_fn, 'wb') as _f:
        _f.write('\n'.join(mtl_lines))

    obj_lines = ['mtllib %s\n' % os.path.split(mtl_fn)[-1]]

    for l in geom:
        newl = 'v  %s' % ('  '.join(map(str, l)))
        obj_lines.append(newl)

    obj_lines.append('\n')
    for l in texCoords:
        newl = 'vt  %s' % ('  '.join(map(str, l)))
        obj_lines.append(newl)

    obj_lines.append('\nusemtl Mat')

    for l1, l2, in zip(coordIndex, texCoordIndex):
        t = ['%d/%d' % (_l1, _l2) for _l1, _l2 in zip(l1, l2)]
        newl = 'f  %s' % '  '.join(t)
        obj_lines.append(newl)

    obj_lines.append('\n')

    obj_fn = fn + '.obj'
    with open(obj_fn, 'wb') as _f:
        _f.write('\n'.join(obj_lines))

