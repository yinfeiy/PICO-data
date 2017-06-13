import os, sys, glob, json


# TODO(yinfeiy) Clean the path to be safe contained instead of depend on other sources

ref_annotype = 'Participants'
ref_path = '/mnt/data/workspace/nlp/PICO-src/med_abst/vis_gts/{0}/'.format(ref_annotype)

# load meta data
meta_data = {}
with open('/mnt/data/workspace/nlp/PICO-src/med_abst/meta_data.txt') as fin:
    for line in fin:
        d = json.loads(line)
        did = d['doc_id']
        meta_data[did] = d

o_path = './output/iframes/'
if not os.path.exists(o_path):
    os.makedirs(o_path)

fns = glob.glob(ref_path+'*.html')
for fn_ref in fns:

    fname = os.path.basename(fn_ref)
    doc_id = fname.replace('.html', '')

    ofn = os.path.join(o_path, fname)
    fo = open(ofn, 'w+')
    fo.write('<h1 align="center">Abstract Doc ID: {0}</h1>\n'.format(doc_id))

    for annotype in ['Participants', 'Intervention', 'Outcome']:
        gt_path = './vis_gts/{0}/'.format(annotype)
        an_path = './vis_annos/{0}/'.format(annotype)
        re_path = './vis_res/{0}/'.format(annotype)
        meta = meta_data[doc_id][annotype]

        fn_gt = os.path.join(gt_path, fname)
        fn_an = os.path.join(an_path, fname)
        fn_re = os.path.join(re_path, fname)
        url_gt = 'https://fling.seas.upenn.edu/~yinfeiy/dynamic/med_abst/' + fn_gt
        url_an = 'https://fling.seas.upenn.edu/~yinfeiy/dynamic/med_abst/' + fn_an
        url_re = 'https://fling.seas.upenn.edu/~yinfeiy/dynamic/med_abst/' + fn_re
        #url_gt = 'file:///Users/yinfei.yang/workspace/nlp/PICO/PICO-src/med_abst/' + fn_gt
        #url_an = 'file:///Users/yinfei.yang/workspace/nlp/PICO/PICO-src/med_abst/' + fn_an

        fo.write('<h2 align="center">Annotype: {0}</h2>\n'.format(annotype))
        fo.write('<h3 align="center"><font color=a52708>')
        fo.write('number of workers: {0}, corr: {1}, prec: {2}, recl: {3}, coverage_tp: {4},\
                coverage_tn: {5}, loss_of_coverage: {6}, impurity: {7}'.format(\
                meta['n_worker'], meta['corr'], meta['prec'], meta['recl'],\
                meta['cov_tp'], meta['cov_tn'], meta['loss_c'], meta['impurity']))
        fo.write('</font></h3>\n')
        fo.write('<table style="width:100%" align="center">\n')
        fo.write('<tr><th>Medical Students</th><th>AMT Workers</th></tr>')
        fo.write('<tr><td align="center">\n')
        fo.write('<iframe src="{0}" height="500" width=99%></iframe>\n'.format(url_gt))
        fo.write('</td><td align="center">\n')
        fo.write('<iframe src="{0}" height="500" width=99%></iframe>\n'.format(url_an))
        fo.write('</td></tr>\n')
        fo.write('</table>\n')
        fo.write('<p>\n')
        fo.write('<iframe src="{0}" frameBorder="0" height=300 width=100%></iframe>\n'.format(url_re))
    fo.close()
