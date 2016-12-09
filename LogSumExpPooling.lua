-- compatible CUDA


local LogSumExpPooling, parent = torch.class('nn.LogSumExpPooling', 'nn.Module')

-- n: number of top instances
function LogSumExpPooling:__init(beta)
    parent.__init(self)
    self.beta = beta or 1
    self.output = torch.Tensor()
end

function LogSumExpPooling:updateOutput(input)
    -- backward compatibility

    if input:dim() == 4 then -- batch
        batchSize = input:size(1)
        numChannels = input:size(2)
        h = input:size(3)
        w = input:size(4)
    elseif input:dim() == 3 then -- image
        batchSize = 1
        numChannels = input:size(1)
        h = input:size(2)
        w = input:size(3)
    else
        print('error in LogSumExpPooling:updateOutput - incorrect input size')
    end

    local xBeta = input:clone():view(batchSize, numChannels, h*w):mul(self.beta)
    local xMax = torch.max(xBeta, 3)
    local xMaxNeg = torch.mul(xMax, -1):expand(batchSize, numChannels, h*w):contiguous()
    local xExp = xBeta:add(xMaxNeg):exp():sum(3):div(h*w):log():div(self.beta)
    local xMaxBeta = torch.div(xMax, self.beta)
    self.output = xExp:add(xMaxBeta)

    if input:dim() == 4 then
        self.output = self.output:view(batchSize, numChannels, 1, 1)
    elseif input:dim() == 3 then
        self.output = self.output:view(numChannels, 1, 1)
    end

    return self.output
end

function LogSumExpPooling:updateGradInput(input, gradOutput)

    if input:dim() == 4 then -- batch
        batchSize = input:size(1)
        numChannels = input:size(2)
        h = input:size(3)
        w = input:size(4)
    elseif input:dim() == 3 then -- image
        batchSize = 1
        numChannels = input:size(1)
        h = input:size(2)
        w = input:size(3)
    else
        print('error in LogSumExpPooling:updateGradInput - incorrect input size')
    end

    local xBeta = input:clone():view(batchSize, numChannels, h*w):mul(self.beta)
    local xMax = torch.max(xBeta, 3)
    local xMaxNeg = torch.mul(xMax, -1):expand(batchSize, numChannels, h*w):contiguous()
    local xExp = xBeta:add(xMaxNeg):exp()
    local xSum = torch.sum(xExp, 3):expand(batchSize, numChannels, h*w):contiguous()
    local xExpSum = torch.cdiv(xExp, xSum)

    self.gradInput = gradOutput:clone():view(batchSize, numChannels, 1):expand(batchSize, numChannels, h*w):contiguous():cmul(xExpSum)

    -- supprime les valeurs < 1e-10
    abs = torch.abs(self.gradInput)
    t = torch.ones(abs:size()):typeAs(input):mul(1e-10)
    mask = torch.lt(abs, t)
    self.gradInput[mask] = 0

    self.gradInput = self.gradInput:view(batchSize, numChannels, h, w)

    if input:dim() == 4 then
        self.gradInput = self.gradInput:view(batchSize, numChannels, h, w)
    elseif input:dim() == 3 then
        self.gradInput = self.gradInput:view(numChannels, h, w)
    end
    return self.gradInput
end

function LogSumExpPooling:empty()
    self.gradInput:resize()
    self.gradInput:storage():resize(0)
    self.output:resize()
    self.output:storage():resize(0)
end

function LogSumExpPooling:__tostring__()
    local s =  string.format('%s(beta=%f)', torch.type(self), self.beta)
    return s
end
