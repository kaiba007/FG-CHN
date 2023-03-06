import torch

def calc_hammingDist(B1, B2):
    q = B2.size(1)  # q=4
    if len(B1.size()) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.size(0)
    map = 0
    if k is None:
        k = retrieval_L.size(0)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.int)
        tsum = torch.sum(gnd)

        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, tsum)
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd, as_tuple=False)[:total].squeeze().type(torch.float32) + 1.0
        map = map + torch.mean(count.to(tindex.device) / tindex)
    map = map / num_query
    return map