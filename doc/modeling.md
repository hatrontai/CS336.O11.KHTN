# Features extraction

Các module feature extaction được sử dụng với mục đích chuyển các đối tượng đầu vào như hình ảnh hay văn bản thành các kiểu dữ liệu có thể thực hiện việc tính toán độ tương đồng thông qua các phép tính toán khoảng cách như [Consine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity). Trong ví dụ này chúng tôi triển khai 2 module feature extraction dựa trên pretrain model [BLIP](https://github.com/salesforce/LAVIS) như sau

- [BLIP](blip.py) : Module này triển khai các function cần thiết cho việc init model, init tokenizer, image encoder dựa trên model [blip_feature_extractor](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip_models/blip_feature_extractor.py)
- [BLIP2](blip2.py): Module này triển khai các function cần thiết cho việc init model, init tokenizer, image encoder dựa trên model [blip2_image_text_matching](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_image_text_matching.py)

Cả 2 module đều trả về một vector có kích thước 256-dim và có thể sử dụng để xây dựng database nhằm mục đích truy xuất thông tin.

# Khởi tạo mô hình và tokenizer

Khởi tạo mô hình và tokenizer cho cả hình ảnh và văn bản:

```python
from blip import get_embed_dim, blip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = init_model('base')
tokenizer = init_tokenizer()
```
# Features Extraction:
## Mã hóa hình ảnh

Để mã hóa hình ảnh, sử dụng hàm `image_encoder`:

```python
from PIL import Image
import os
#Giả sử image_name là đường dẫn của ảnh bạn muốn encode
img = Image.open(image_name)
img_emb = image_encoder(img, model, vis_processors, device)
```

Output sample:
```python
{"image_name":"2715746315.jpg","img_emb":[0.0117941108,-0.0189110469,-0.0531733297,0.0991272256,-0.1151893213,0.0583473891,-0.0309707709,0.0464132689,-0.0822054073,-0.0457523465,-0.0244460385,0.0156382136,0.0524512157,-0.0718549341,0.1064621508,0.022221053,0.0036621091,-0.0445459224,0.0219192877,0.0572955124,-0.0357077122,-0.015450621,0.1487883776,0.0765488446,0.1197176352,-0.0063879509,0.070185408,0.1313013434,-0.0075475858,-0.0342122354,-0.0333532803,0.0025794783,-0.1334939599,-0.0288509149,0.0156752169,-0.0622369796,0.0459192842,0.0004274009,-0.0344448686,0.0093030687,0.0493152514,0.0302130748,-0.0073410892,0.0046094377,-0.0986717343,0.0122982524,0.0452497825,-0.1232430041,0.0493150428,-0.064748086,0.0354732722,-0.0103302635,-0.0086322557,0.0294114258,-0.015177385,-0.0322560854,-0.0184529666,0.0092848185,-0.1534446031,-0.0337911248,0.0432127155,-0.0064906199,-0.0464992188,-0.0792911798,0.0748046264,0.0728881434,-0.0129501987,0.0101490049,-0.000706101,-0.0058353264,-0.0958959982,0.0543614961,0.0623242259,0.0094372323,0.0501777083,0.0269158836,-0.0782485232,0.2011345029,0.02753569,0.0114157209,-0.0161079783,0.0793774053,-0.0869617537,-0.0051848213,-0.0953810513,0.0172841474,-0.0522383191,0.0235095825,-0.0453367941,-0.0383974463,0.0427605808,-0.0176272634,-0.1436238885,0.022801714,0.0395614542,-0.0265461486,-0.0632031187,-0.1013688073,-0.0491973795,-0.0163615737,0.0047282297,-0.0891227648,0.0087028379,0.0402028449,-0.0352190211,0.0088042058,-0.0567847379,0.040110603,0.0603878759,0.0300702397,0.0250039361,-0.0252612885,-0.0960186496,-0.0209329948,-0.0136140287,0.0764810368,0.0194041636,0.0801608413,0.0130641246,-0.0671623647,-0.1071168482,-0.052734293,-0.0922797248,0.0315520018,-0.0963769332,-0.026915228,0.0425261408,0.1044192538,0.0456072576,0.0145767936,0.0102903629,-0.0483332947,0.0610527731,-0.0627901852,-0.0999919996,-0.0107981209,0.0842787698,0.0372397527,0.0585491769,0.0345280319,-0.0678416267,0.0263785496,0.0070446804,-0.0949658975,-0.1484363079,0.0715348572,-0.0727064759,0.0061068386,-0.0823842883,-0.0201122779,0.1465644091,-0.0772738978,-0.0238473658,0.0399917215,-0.0335317068,-0.0046041785,0.0131542264,0.0063112364,-0.0150651904,-0.0748230442,0.0276285075,-0.0425018109,-0.0226072744,0.0406122804,0.0761609375,0.0242102668,-0.0385664999,0.0659633279,0.0128254285,0.0321631357,0.0336442143,0.0587616004,-0.0549379773,-0.0287723355,0.0127805248,0.0720557123,0.0926281586,0.0047310465,-0.0755295455,-0.0660084635,-0.0346123949,-0.0757981315,0.0747351497,-0.0322368369,0.113366358,0.0818228647,0.004621482,-0.028623037,0.0412724763,0.0141565604,0.0299777258,0.0003597309,0.0308809653,0.0443706848,-0.0752513483,0.0289211906,0.0160519406,-0.0119854249,-0.0521475635,0.0254557673,-0.1507754177,-0.0020484296,-0.0253612678,-0.0219172426,0.002797219,0.0108646052,-0.0327735879,-0.0286525413,0.0118680652,0.1145852655,0.0777823627,0.025314603,-0.0231918301,-0.1479450911,0.0325094014,0.0275352988,-0.0376665406,-0.0308757517,-0.0519636273,-0.0308052152,0.0446967892,-0.1774981618,0.0689199269,0.075837411,-0.0002923989,-0.0348664261,-0.0318561085,0.0396137014,0.0310923737,0.0568220653,-0.0185633004,0.0690071359,0.0218476597,-0.0746226013,-0.0638026297,-0.0250773765,0.0547785908,-0.106685698,0.0653466284,-0.0446078144,-0.0333743244,0.0724673942,0.0986475497,-0.0097764432,-0.120769307,-0.0531274565,-0.0425590612,0.071892418,0.0876139253,-0.0549242198,0.0160252955,-0.0572688915,0.1313213259,-0.1748223454,-0.080285579,0.11920017]}
```

## Mã hóa văn bản

Mã hóa văn bản sử dụng hàm `text_encoder`:

```python
# Giả sử `caption` là một chuỗi văn bản
text_emb = text_encoder(caption, model, tokenizer, txt_processors, device)
```

Output sample:
```python
{"text":" a man wearing a red jacket sitting on a bench next to various camping items","text_emb":[-0.1354810894,0.1071185693,-0.0784452707,-0.0051497594,0.1006783992,-0.0196148083,0.0308040902,0.0600677244,-0.0661319345,-0.1034791097,0.0295604039,-0.0943820924,0.0258751251,-0.0065143118,0.048675321,0.1043668017,0.0785931572,-0.0150029445,-0.0378446914,0.004402075,0.0733426586,0.0389816873,-0.0850672796,-0.0062821233,0.0296697915,-0.0557634756,-0.0359775648,0.0621644892,0.0101905018,0.0809092,0.0107551292,0.0955846682,-0.0019246206,-0.1234848052,0.0146287186,0.0407720208,0.0494324379,0.0251427144,-0.0729953051,0.0533342436,-0.0008606671,-0.0063491683,-0.0495416746,-0.0002539918,0.0252626836,0.0244860072,0.0368785039,-0.0923363566,-0.0154104428,-0.1293037534,0.0056978208,0.0819168985,0.0470197499,0.1832970381,0.163348332,-0.0392420925,-0.0130341845,-0.0425376259,-0.0530539267,-0.0348570198,-0.0451212227,-0.0218534339,0.110000968,-0.0274976511,0.0581795312,0.0869170427,-0.0093112914,-0.051274091,-0.0022846293,0.0410992652,0.0940619111,-0.004297941,-0.0026053181,-0.0569229312,0.0443793237,-0.0256765764,0.0634455979,-0.0376053713,-0.0494387969,0.0397577547,0.0540629588,-0.0513045937,-0.1631050557,0.069930926,0.0035641077,-0.0273034237,-0.0533498228,-0.0214694329,0.0161216147,0.0031206207,-0.0380589366,-0.0633520633,0.0032757143,0.0757682845,-0.0388415791,0.0061146175,-0.0310624298,-0.1712888479,-0.0247960202,-0.0546058714,-0.0162009671,-0.0503657311,0.0457477719,0.0685178265,-0.0560199469,0.0477728993,0.0209318213,-0.0198890436,-0.0111354515,-0.049411431,0.0474371761,-0.064535968,0.019443946,-0.0398051701,0.0363844149,0.0445033461,-0.1031457186,0.1007297114,-0.0039237798,-0.0722766295,-0.0834626257,0.0121162198,-0.0421609394,-0.0806439072,-0.0924008787,-0.1292526722,-0.0436791405,0.0528297164,0.0200978164,0.0718566999,0.027651649,0.0647834688,0.08407031,0.0261327047,0.0480939224,0.1199071035,0.0350221656,-0.0416413769,0.0092657283,-0.0081748301,-0.087462835,0.0456500575,-0.0592514053,-0.0302098449,0.0133185182,-0.0963640958,0.054842107,0.1835112572,0.0215675104,-0.0014639492,0.0094935643,0.0509745926,0.0114533743,-0.0616363697,-0.0018645922,-0.0092656445,0.0337426104,-0.0817203447,-0.0191729162,-0.0070789629,0.0097877057,0.024571985,-0.0128598912,0.072874032,-0.0425730012,-0.0601729266,0.036712084,-0.0532311387,-0.0963106304,-0.0259603523,-0.1275657266,-0.0183546413,-0.0056884401,0.06645789,-0.0306243077,0.0392024107,-0.0281947777,-0.1039740518,-0.0718720481,-0.0816787481,-0.0456937663,-0.0647952408,-0.0016371103,-0.0155811217,-0.0755044892,-0.0095721781,-0.0586225241,0.1889934987,-0.0329983532,0.0028412242,-0.0653807297,-0.0005356898,-0.0817067772,0.0854337588,-0.0752266273,-0.1213191748,-0.0236229654,-0.0521311462,0.0039025287,-0.0223065261,0.0015729466,0.0491905175,-0.002567067,-0.0555180684,-0.0176982842,-0.0490777493,0.0046092281,0.0867602006,-0.0238536596,-0.0757027939,0.0654661804,0.0005102981,-0.0181402359,0.0376366749,-0.0421681739,0.0203027315,-0.0454545803,0.0336169414,-0.082916297,-0.0009641394,0.0737567544,-0.1301347464,-0.0228111446,-0.0427748114,-0.1599926651,0.0035070672,0.0620546415,0.0222923439,-0.0093585243,0.0340745933,0.0344987027,0.0421209969,0.0300749112,0.0386573374,-0.0127013708,0.0024380994,0.0654817298,-0.1767539531,-0.0576897822,-0.0843090564,-0.1029165015,0.0483418405,0.0961887389,0.0483225435,-0.0850682333,0.0913516209,-0.0025616526,-0.078044109,-0.0042927973,-0.0014180822,-0.0156627819,0.0057553379,-0.0174222495,-0.059247274,-0.0382565223,-0.0419340022]}
```