-- compatible CUDA
require 'torch'
require 'nn'

local GlobalkMaxPooling, parent = torch.class('nn.GlobalkMaxPooling', 'nn.Module')

function GlobalkMaxPooling:__init(nMax)
    parent.__init(self)
    self.nMax = nMax or 1
    self.output = torch.Tensor()
    self.indices = torch.Tensor()
end

function GlobalkMaxPooling:updateOutput(input)

    local batchSize = 0
    local numChannels = 0
    local h = 0
    local w = 0

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
        print('error in GlobalkMaxPooling:updateOutput - incorrect input size')
    end

    local nMax = self.nMax
    if nMax <= 0 then
        nMax = math.max(1, math.floor(h * w))
    elseif nMax < 1 then
        nMax = math.max(1, math.floor(nMax * h * w))
    end

    self.output:typeAs(input):resize(batchSize, numChannels, 1, 1)
    local x = input:view(batchSize, numChannels, h*w)

    -- sort scores by decreasing order
    local scoreSorted, indices = torch.sort(x, x:size():size(), true)

    -- compute top max
    self.indices = indices[{{},{},{1,nMax}}]
    torch.sum(self.output, scoreSorted[{{},{},{1,nMax}}], 3)
    self.output:div(nMax)

    if input:dim() == 4 then
        self.output = self.output:view(batchSize, numChannels, 1, 1)
    elseif input:dim() == 3 then
        self.output = self.output:view(numChannels, 1, 1)
    end

    return self.output
end

function GlobalkMaxPooling:updateGradInput(input, gradOutput)

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
        print('error in GlobalkMaxPooling:updateGradInput - incorrect input size')
    end

    local nMax = self.nMax
    if nMax <= 0 then
        nMax = math.max(1, math.floor(h * w))
    elseif nMax < 1 then
        nMax = math.max(1, math.floor(nMax * h * w))
    end

    local yMax = torch.expand(gradOutput:clone():view(batchSize, numChannels, 1), batchSize, numChannels, nMax)
    local z = torch.zeros(batchSize, numChannels, h*w):typeAs(input)
    self.gradInput = z:scatter(3, self.indices, yMax):div(nMax)

    if input:dim() == 4 then
        self.gradInput = self.gradInput:view(batchSize, numChannels, h, w)
    elseif input:dim() == 3 then
        self.gradInput = self.gradInput:view(numChannels, h, w)
    end


    return self.gradInput
end

function GlobalkMaxPooling:empty()
    self.gradInput:resize()
    self.gradInput:storage():resize(0)
    self.output:resize()
    self.output:storage():resize(0)
    self.indices:resize()
    self.indices:storage():resize(0)
end

function GlobalkMaxPooling:__tostring__()
    local s =  string.format('%s(nMax=%d)', torch.type(self), self.nMax)
    return s
end
