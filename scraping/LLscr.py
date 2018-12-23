import scr2

lllist = ['高坂穂乃果', '南ことり', '園田海未', '小泉花陽', '星空凛', '西木野真姫', '矢澤にこ', '絢瀬絵里', '東條希']
classes = ['honoka', 'kotori', 'umi', 'hanayo', 'rin', 'maki', 'nico', 'eli', 'nozomi']  # 保存名英語でないと後々面倒

others = ["きんモザ", "てーきゅう", "ラブライブサンシャイン", "ごちうさ", "りゅうおうのおしごと!"]
others2 = ["アイマス", "sideM", "アイカツ", "けいおん！"]
search_list = others + others2

if __name__ == '__main__':
    for i in range(len(search_list)):
        scr2.scrape(search_list[i]+"　キャプ", 1, search_list[i])
