import numpy as np
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 各フィルターの要素（マス目）を二つのfor文で走らせる。
    # y_max, x_maxはそれぞれy, xの移動する最大の位置（ゴール）
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            # col[N, C, filter_h, filer_w, out_h, out_w]
            # img[N, C, padded_H, padded_W]
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # i.e.> フィルターの`(y, x)`成分が各ステップで相手となる入力画像の各要素（`y:y_max:stride, x:x_max:stride`で指定）を取得する

    # colの場合、（transposeした後）reshapeメソッドはcol[0, 0, 0, 0, 0, 1]から順に並べていき、その並べたものから順にピックアップしていくことで、reshapeで指定した形状の配列を生成する。
    # 今回は、colの第1, 2, 3次元目をすべて1行（横向き）にフラット化したもの縦に重ねていく。
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
    # col.transpose(N, out_h, out_w, channel, filter_h, filter_w)
    # `(N, out_h, out_w, C, filter_h, filter_w)`という形状になる。

    # 各行、最初にfilter_wの情報が格納されていく（整形されていく）。
    # 格納されるイメージは次の通り↓。
    # i.e.> 「最初は0個目のデータにおける出力画像の`(0, 0)`のピクセルにおける積和演算について、フィルターの0行目の0列目のデータから1列目、2列目、...、filter_w列目までが、整形後の`col`を行方向に、順に格納されていく。」
    # > 「そうしたら、filter_h = 0におけるfilter_wが最後まで行ったから、filter_h = 1に繰り上がり、再度各列が続きから格納されていく。」
    # > つまりは、かけ算の意味の定義（掛けられる数 * 掛ける数）のイメージで整形後の`col`の各行（各行の要素数はC*filter_h*filter_w）が出来上がっていく。
    # > `filter_w`分のセットが`filter_h`だけあり、さらにそれが`C`個あるようなイメージ。`C`個の箱の中に、さらに`filter_h`個の箱があり、それらの箱の中にはそれぞれ`filter_w`個のケーキが入っているようなこと。

    # なんか、この操作って足し算の繰り上がりみたいだね。


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape    # 入力画像の形状を記憶（データ数、チャンネル数、高さ、幅）
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)   # 入力されてきたcol(形状が`(N*out_h*out_w、C*FH*FW)`)を`(N, out_h, out_w, C, filter_h, filter_w)`という形状に整え、`transpose`によって、im2colの`col`の定義と同様の形状にする。
    # col2imの引数`col`が想定しているのは、二次元にreshapeした後の`col`（`(N*out_h*out_w, C*FH*FW)`）であることに注意。
    # だから、im2colの出力結果がそのまま引数`col`に入ることを想定している。

    # ↓AIのまとめ
    # 引数として入力されてくるのは、逆伝播によって上流から流れてきた「2次元行列の勾配データ（dcol）」である。
    # これを im2col の逆手順で復元していく。
    # まず reshape によって (N, out_h, out_w, C, filter_h, filter_w) にほぐし、
    # さらに transpose(0, 3, 4, 5, 1, 2) を行うことで、
    # im2col の時に作った「6次元の整理タンス（N, C, FH, FW, OH, OW）」と全く同じ形状に復元する。

    # 次回はここ（↓）から（2026-02-25）

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))  # `+ stride - 1`をしているのは、yからy_max（y_max - 1）までのインデックスに値を代入するとき、`y_max`がとりうる最大値が`H + 2*pad + stride - 1`で抑えられる。そのため`H + 2*pad`という、本来の画像サイズに`stride - 1`を足すことで、バッファを作ることができる。

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            # imgとcolの各次元の意味
            # img[データ数, チャンネル数, 画像の高さ（拡張）, 画像の幅（拡張）]
            # col[データ数, チャンネル数, フィルターの高さ, フィルターの幅, 出力画像の高さ, 出力画像の幅]

    # imgによって生成したすべて白紙（0）の画像の4次元配列（データ数は N、チャンネルが C で、高さと幅はパディングと安全バッファを含んだ最大サイズ）の、任意のデータ数の任意のチャンネルの高さの次元において、stride間隔で y から y_max の手前までの要素（幅についても同様）に対し、6次元配列colの任意のデータ数の任意のチャンネルにおける (y, x) 番目のフィルターと対応している出力画像の任意の要素が加算される（足し込まれる）


    return img[:, :, pad:H + pad, pad:W + pad]  # 先ほど作ったバッファを、パディングとともに削るために「`pad:H+pad`」をする。そうすると、その要素数は`H - pad + pad = H`となり、もとの画像の高さ（幅も同様）になる。さらに、`pad`から始めることで、パディングの分も削ることができる。疑心暗鬼になったら具体例やこのスライス操作をイメージせよ。