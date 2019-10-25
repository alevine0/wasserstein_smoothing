import torch
import math
import scipy.stats
import numpy
import statsmodels.stats.proportion
# note: output is softmaxed (pre-log) score. Use nn.NLLLoss(torch.log(out))
def laplace_smooth_forward(batch, net, num_samples, stdev):
	m = torch.distributions.laplace.Laplace(torch.tensor([0.0]).cuda(), torch.tensor([stdev/(2 ** 0.5)]).cuda())
	batch_size = batch.shape[0]
	normalization = batch.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
	batch = batch/normalization
	expanded = batch.repeat_interleave(num_samples,0) # shape: batch*num_samples, etc
	noise = m.sample(sample_shape=expanded.shape).squeeze(dim=-1)
	expanded += noise
	soft = net(expanded*normalization.repeat_interleave(num_samples,0))
	votes = soft.max(1)[1]
	hard = torch.zeros(soft.shape).cuda()
	hard.scatter_(1,votes.unsqueeze(1),1)
	return hard.reshape((batch.shape[0],num_samples,) + hard.shape[1:]).mean(dim=1)


# note: output is softmaxed (pre-log) score. Use nn.NLLLoss(torch.log(out))
def wass_smooth_forward(batch, net, num_samples, stdev, channel_denorm = None, channel_norm = None):
	m = torch.distributions.laplace.Laplace(torch.tensor([0.0]).cuda(), torch.tensor([stdev/(2 ** 0.5)]).cuda())
	batch_size = batch.shape[0]
	if (channel_denorm):
		batch = channel_denorm(batch)
	normalization = batch.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
	batch = batch/normalization
	expanded = batch.repeat_interleave(num_samples,0) # shape: batch*num_samples, etc
	flow_vert = m.sample(sample_shape=expanded.shape[0:-2]+(expanded.shape[-2]-1,)+(expanded.shape[-1],)).squeeze(dim=-1)
	flow_horiz = m.sample(sample_shape=expanded.shape[0:-2]+(expanded.shape[-2],)+(expanded.shape[-1]-1,)).squeeze(dim=-1)
	expanded[:,:,:-1,:] += flow_vert
	expanded[:,:,1:,:] -= flow_vert
	expanded[:,:,:,:-1] += flow_horiz
	expanded[:,:,:,1:] -= flow_horiz
	if (channel_norm):
		soft = net(channel_norm(expanded*normalization.repeat_interleave(num_samples,0)))
		votes = soft.max(1)[1]
		hard = torch.zeros(soft.shape).cuda()
		hard.scatter_(1,votes.unsqueeze(1),1)
		out = hard.reshape((batch.shape[0],num_samples,) + hard.shape[1:]).mean(dim=1)
	else:
		soft = net(expanded*normalization.repeat_interleave(num_samples,0))
		votes = soft.max(1)[1]
		hard = torch.zeros(soft.shape).cuda()
		hard.scatter_(1,votes.unsqueeze(1),1)
		out = hard.reshape((batch.shape[0],num_samples,) + hard.shape[1:]).mean(dim=1)
	return out

def soft_wass_smooth_forward(batch, net, num_samples, stdev, channel_denorm = None, channel_norm = None):
	m = torch.distributions.laplace.Laplace(torch.tensor([0.0]).cuda(), torch.tensor([stdev/(2 ** 0.5)]).cuda())
	batch_size = batch.shape[0]
	if (channel_denorm):
		batch = channel_denorm(batch)
	normalization = batch.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
	batch = batch/normalization
	expanded = batch.repeat_interleave(num_samples,0) # shape: batch*num_samples, etc
	flow_vert = m.sample(sample_shape=expanded.shape[0:-2]+(expanded.shape[-2]-1,)+(expanded.shape[-1],)).squeeze(dim=-1)
	flow_horiz = m.sample(sample_shape=expanded.shape[0:-2]+(expanded.shape[-2],)+(expanded.shape[-1]-1,)).squeeze(dim=-1)
	expanded[:,:,:-1,:] += flow_vert
	expanded[:,:,1:,:] -= flow_vert
	expanded[:,:,:,:-1] += flow_horiz
	expanded[:,:,:,1:] -= flow_horiz
	if (channel_norm):
		soft = torch.nn.functional.softmax(net(channel_norm(expanded*normalization.repeat_interleave(num_samples,0))), dim=-1)
		out = soft.reshape((batch.shape[0],num_samples,) + soft.shape[1:]).mean(dim=1)
	else:
		soft = torch.nn.functional.softmax(net(expanded*normalization.repeat_interleave(num_samples,0)), dim=-1)
		out = soft.reshape((batch.shape[0],num_samples,) + soft.shape[1:]).mean(dim=1)
	return out

def wass_smooth_forward_train(batch, net, stdev, channel_denorm = None, channel_norm = None):
	m = torch.distributions.laplace.Laplace(torch.tensor([0.0]).cuda(), torch.tensor([stdev/(2 ** 0.5)]).cuda())
	batch_size = batch.shape[0]
	if (channel_denorm):
		batch = channel_denorm(batch)
	normalization = batch.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
	batch = batch/normalization
	flow_vert = m.sample(sample_shape=batch.shape[1:-2]+(batch.shape[-2]-1,)+(batch.shape[-1],)).squeeze(dim=-1)
	flow_horiz = m.sample(sample_shape=batch.shape[1:-2]+(batch.shape[-2],)+(batch.shape[-1]-1,)).squeeze(dim=-1)
	batch[:,:,:-1,:] += flow_vert
	batch[:,:,1:,:] -= flow_vert
	batch[:,:,:,:-1] += flow_horiz
	batch[:,:,:,1:] -= flow_horiz
	if (channel_norm):
		out =  torch.nn.functional.softmax(net(channel_norm(batch*normalization)))
	else:
		out = torch.nn.functional.softmax(net(batch*normalization))
	return out

def laplace_smooth_forward_train(batch, net, stdev):
	m = torch.distributions.laplace.Laplace(torch.tensor([0.0]).cuda(), torch.tensor([stdev/(2 ** 0.5)]).cuda())
	batch_size = batch.shape[0]
	normalization = batch.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
	batch = batch/normalization
	noise = m.sample(sample_shape=batch[0].shape).unsqueeze(dim=0).squeeze(-1)
	#print(batch.shape)
	#print(noise.shape)
	batch += noise
	return torch.nn.functional.softmax(net(batch*normalization))


def lc_bound(k, n ,alpha):
	return statsmodels.stats.proportion.proportion_confint(k, n, alpha=2*alpha, method="beta")[0]

#Accounting for L1 vs L2 metric in image
def wass_smooth_certify(batch, labels, net, alpha, stdev, num_samples_select, num_samples_bound, channel_norm = None ):
	guesses = wass_smooth_forward(batch, net, num_samples_select, stdev,channel_norm=channel_norm).max(1)[1]
	bound_scores = wass_smooth_forward(batch, net, num_samples_bound, stdev,channel_norm=channel_norm)
	bound_selected_scores = torch.gather(bound_scores,1,guesses.unsqueeze(1)).squeeze(0)
	bound_selected_scores = torch.tensor(lc_bound((bound_selected_scores*num_samples_bound).cpu().numpy(),num_samples_bound,alpha)).cuda()
	radii = torch.clamp((torch.log(bound_selected_scores)-torch.log(1.-bound_selected_scores))*stdev/(2 ** 1.5), min=0)
	radii = radii/(2 ** 0.5)
	radii[guesses != labels] = -1
	return radii

def wass_predict(batch, net ,stdev, num_samples, alpha, channel_norm = None ):
	scores = wass_smooth_forward(batch, net, num_samples, stdev,channel_norm=channel_norm)
	toptwo = torch.topk(scores.cpu(),2,sorted=True)
	toptwoidx = toptwo[1]
	toptwocounts = toptwo[0]*num_samples
	out = -1* torch.ones(batch.shape[0], dtype = torch.long)
	tests = numpy.array([scipy.stats.binom_test(toptwocounts[idx,0],toptwocounts[idx,0]+toptwocounts[idx,1], .5) for idx in range(batch.shape[0])])
	out[tests <= alpha] = toptwoidx[tests <= alpha][:,0]
	return out

def laplace_predict(batch, net ,stdev, num_samples, alpha):
	scores = laplace_smooth_forward(batch, net, num_samples, stdev)
	toptwo = torch.topk(scores.cpu(),2,sorted=True)
	toptwoidx = toptwo[1]
	toptwocounts = toptwo[0]*num_samples
	out = -1* torch.ones(batch.shape[0], dtype = torch.long)
	tests = numpy.array([scipy.stats.binom_test(toptwocounts[idx,0],toptwocounts[idx,0]+toptwocounts[idx,1], .5) for idx in range(batch.shape[0])])
	out[tests <= alpha] = toptwoidx[tests <= alpha][:,0]
	return out

# Note: to certify a wasserstien perturbation via laplace smoothing,
#	it must be certified to twice the wasserstein radius
def laplace_smooth_certify(batch, labels, net, alpha, stdev, num_samples_select, num_samples_bound ):
	guesses = laplace_smooth_forward(batch, net, num_samples_select, stdev).max(1)[1]
	bound_scores = laplace_smooth_forward(batch, net, num_samples_bound, stdev)
	bound_selected_scores = torch.gather(bound_scores,1,guesses.unsqueeze(1)).squeeze(0)
	bound_selected_scores = torch.tensor(lc_bound((bound_selected_scores*num_samples_bound).cpu().numpy(),num_samples_bound,alpha)).cuda()
	radii = torch.clamp((torch.log(bound_selected_scores)-torch.log(1.-bound_selected_scores))*stdev/(2 ** 1.5), min=0)
	radii = radii/(2.)
	radii[guesses != labels] = -1
	return radii
