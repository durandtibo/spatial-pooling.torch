-- compatible CUDA
require 'torch'
require 'nn'

local WeldonPooling, parent = torch.class('nn.WeldonPooling', 'nn.Module')

function WeldonPooling:__init(nMax, nMin)
    parent.__init(self)
    self.nMax = nMax or 1
    self.nMin = nMin or self.nMax

    self.output = torch.Tensor()
    self.indicesMax = torch.Tensor()
    self.indicesMin = torch.Tensor()
end

function WeldonPooling:updateOutput(input)

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
        print('error in WeldonPooling:updateOutput - incorrect input size')
    end

    local nMax = self.nMax
    if nMax <= 0 then
        nMax = 0
    elseif nMax < 1 then
        nMax = math.max(1, math.floor(nMax * h * w))
    end
    local nMin = self.nMin
    if nMin <= 0 then
        nMin = 0
    elseif nMin < 1 then
        nMin = math.max(1, math.floor(nMin * h * w))
    end

    self.output:typeAs(input):resize(batchSize, numChannels, 1, 1)
    local x = input:view(batchSize, numChannels, h*w)

    -- sort scores by decreasing order
    local scoreSorted, indices = torch.sort(x, x:size():size(), true)

    -- compute top max
    self.indicesMax = indices[{{},{},{1,nMax}}]
    torch.sum(self.output, scoreSorted[{{},{},{1,nMax}}], 3)
    self.output:div(nMax)

    -- compute top min
    if nMin > 0 then
        self.indicesMin = indices[{{},{},{h*w-nMin+1,h*w}}]
        local yMin = torch.sum(scoreSorted[{{},{},{h*w-nMin+1,h*w}}], 3):div(nMin)
        torch.add(self.output, self.output, yMin)
    end

    if input:dim() == 4 then
        self.output = self.output:view(batchSize, numChannels, 1, 1)
    elseif input:dim() == 3 then
        self.output = self.output:view(numChannels, 1, 1)
    end

    return self.output
end

function WeldonPooling:updateGradInput(input, gradOutput)

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
        print('error in WeldonPooling:updateGradInput - incorrect input size')
    end

    local nMax = self.nMax
    if nMax <= 0 then
        nMax = 0
    elseif nMax < 1 then
        nMax = math.max(1, math.floor(nMax * h * w))
    end
    local nMin = self.nMin
    if nMin <= 0 then
        nMin = 0
    elseif nMin < 1 then
        nMin = math.max(1, math.floor(nMin * h * w))
    end

    local yMax = torch.expand(gradOutput:clone():view(batchSize, numChannels, 1), batchSize, numChannels, nMax)
    local z = torch.zeros(batchSize, numChannels, h*w):typeAs(input)
    z:scatter(3, self.indicesMax, yMax):div(nMax)

    if nMin > 0 then
        local yMin = torch.expand(gradOutput:clone():view(batchSize, numChannels, 1):div(nMin), batchSize, numChannels, nMin)
        self.gradInput = z:scatter(3, self.indicesMin, yMin):view(batchSize, numChannels, h, w)
    else
        self.gradInput = z:view(batchSize, numChannels, h, w)
    end

    if input:dim() == 3 then
        self.gradInput = self.gradInput:view(numChannels, h, w)
    end

    return self.gradInput
end

function WeldonPooling:empty()
    self.gradInput:resize()
    self.gradInput:storage():resize(0)
    self.output:resize()
    self.output:storage():resize(0)
    self.indicesMax:resize()
    self.indicesMax:storage():resize(0)
    self.indicesMin:resize()
    self.indicesMin:storage():resize(0)
end

function WeldonPooling:__tostring__()
    local s =  string.format('%s(nMax=%d,nMin=%d)', torch.type(self), self.nMax, self.nMin)
    return s
end
